import os
import numpy as np
from tqdm import tqdm

import torch
from torchdiffeq import odeint
from torch import Tensor
from torch.nn import MSELoss 
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, TensorDataset

from models.triangular import TriangularFineTune 
from models.JiT import JiTUncond_B_8, JiT_XS_8
import argparse

torch.manual_seed(159753)
np.random.seed(159753)
from solvers.torch_eit_fem_solver.utils import dtn_from_sigma
from solvers.torch_eit_fem_solver.fem import Mesh, V_h
import scipy.io as sio

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch._dynamo.config.dynamic_shapes = True
torch._dynamo.config.recompile_limit = 64

EPS = 5e-2
SIGMA_MIN = 1e-2
DTN_STEPS = 10
DTN_WEIGHT = 16
PINN_BATCH = 4
TRAIN_X_MIN = None
TRAIN_X_MAX = None
TRAIN_Y_MIN = None
TRAIN_Y_MAX = None

P_STD = 0.8
P_MEAN = -0.8


@torch.compile
def interpolate_xt(t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
    t = t[:, None, None, None]
    return t * x1 + (1 - t) * x0


@torch.compile
def step(t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
    t = t[:, None, None, None]
    mu = t * x1
    sigma = 1 - (1 - SIGMA_MIN) * t
    return sigma * x0 + mu


@torch.compile
def target(t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
    return x1 - (1 - SIGMA_MIN) * x0


@torch.compile
def sample_t(n: int, device=None):
    z = torch.randn(n, device=device) * P_STD + P_MEAN
    return torch.sigmoid(z)


def v_from_x_pred(model: TriangularFineTune, z: Tensor, t: Tensor, x_cond: Tensor) -> Tensor:
    t_batch = t.expand(z.shape[0])
    t_broadcast = t_batch[:, None, None, None]
    with torch.autocast(device_type=t.device.type, dtype=torch.bfloat16):
        x_pred = model(z, t_batch)
    denom = torch.clamp(1 - (1 - SIGMA_MIN) * t_broadcast, min=EPS)
    v_pred = x_pred - (1 - SIGMA_MIN) * (z - t_broadcast * x_pred) / denom
    v_pred[:, :1] = 0.0
    return v_pred


def sample_sigma_from_noise(
    model: TriangularFineTune,
    x_cond: Tensor,
) -> Tensor:
    if x_cond.ndim == 3:
        x_cond = x_cond.unsqueeze(1)
    x0 = torch.randn_like(x_cond)
    z0 = torch.cat([x_cond, x0], dim=1)
    timesteps = torch.linspace(0.0, 1.0, steps=DTN_STEPS, device=x_cond.device)
    pred = odeint(
        func=lambda t, z: v_from_x_pred(model, z, t, x_cond),
        t=timesteps,
        y0=z0,
        method="heun2",
        atol=1e-5,
        rtol=1e-5,
    )[-1]
    return pred[:, 1:2]


def dtn_loss_from_sigma(
    sigma_pred: Tensor,
    x_cond: Tensor,
    mesh: Mesh,
    v_h: V_h,
    background: Tensor,
) -> Tensor:
    if x_cond.ndim == 3:
        x_cond = x_cond.unsqueeze(1)
    background = background.to(sigma_pred.device)
    with torch.autocast(device_type=sigma_pred.device.type, enabled=False):
        sigma_phys = (0.5 * (sigma_pred + 1.0) * (TRAIN_Y_MAX - TRAIN_Y_MIN) + TRAIN_Y_MIN).float()
        dtn_preds = []
        for i in range(sigma_phys.shape[0]):
            dtn = dtn_from_sigma(
                sigma_vec=sigma_phys[i, 0],
                v_h=v_h,
                mesh=mesh,
                img_size=128,
                device=sigma_phys.device,
            )
            dtn = dtn / background
            dtn = 2.0 * (dtn - TRAIN_X_MIN) / (TRAIN_X_MAX - TRAIN_X_MIN + 1e-12) - 1.0
            dtn_preds.append(dtn)
        dtn_pred = torch.stack(dtn_preds).unsqueeze(1)
    return MSELoss()(dtn_pred.to(x_cond.dtype), x_cond)

def get_loss_fn(
    model: TriangularFineTune,
    mesh: Mesh,
    v_h: V_h,
    background: Tensor,
):
    def loss_fn(x: Tensor) -> Tensor:
        t = sample_t(x.shape[0], device=x.device)
        t_broadcast = t[:, None, None, None]

        x1 = x.float()
        x0 = torch.randn_like(x1)
        z_t = step(t, x0, x1)
        v = target(t, x0, x1)

        x_cond = x[:, :1]

        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            x_pred = model(z_t.to(dtype=x.dtype), t)

        sigma = torch.clamp(1 - (1 - SIGMA_MIN) * t_broadcast, min=EPS)
        v_pred = x_pred - (1 - SIGMA_MIN) * (z_t - t_broadcast * x_pred) / sigma
        loss = MSELoss()(v, v_pred)
        pinn_bs = min(PINN_BATCH, x_cond.shape[0])
        pinn_idx = torch.randperm(x_cond.shape[0], device=x_cond.device)[:pinn_bs]
        x_cond_pinn = x_cond[pinn_idx]
        sigma_pred = sample_sigma_from_noise(model, x_cond_pinn)
        dtn_loss = dtn_loss_from_sigma(
            sigma_pred=sigma_pred,
            x_cond=x_cond_pinn,
            mesh=mesh,
            v_h=v_h,
            background=background,
        )
        loss = loss + DTN_WEIGHT * dtn_loss
        return loss

    return loss_fn

def get_background(mesh_file, device='cuda'):
    mat_contents = sio.loadmat(mesh_file)

    p = torch.tensor(mat_contents['p'], dtype=torch.float64).to(device)
    t = torch.tensor(mat_contents['t']-1, dtype=torch.long).to(device)
    vol_idx = torch.tensor(mat_contents['vol_idx'].reshape((-1,))-1, dtype=torch.long).to(device)
    bdy_idx = torch.tensor(mat_contents['bdy_idx'].reshape((-1,))-1, dtype=torch.long).to(device)

    mesh = Mesh(p, t, bdy_idx, vol_idx, device=device)
    v_h = V_h(mesh)

    # get the dtn map
    dtn_background = dtn_from_sigma(
        sigma_vec=torch.ones(128, 128, device=device),
        v_h=v_h,
        mesh=mesh,
        img_size=128,
        device=device,
    )
    
    return mesh, v_h, dtn_background 


def get_dtn_from_y(
    y: torch.Tensor,
    *,
    v_h: V_h,
    mesh: Mesh,
    background: torch.Tensor,
    img_size: int = 128,
) -> torch.Tensor:
    if y.ndim == 4 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 3:
        raise ValueError(f"Expected y with shape (B,H,W) or (B,1,H,W), got {tuple(y.shape)}")

    dtns = []
    for i in range(y.shape[0]):
        dtn = dtn_from_sigma(
            sigma_vec=y[i],
            v_h=v_h,
            mesh=mesh,
            img_size=img_size,
            device=y.device,
        )
        dtns.append(dtn)
    dtn = torch.stack(dtns, dim=0)
    return dtn / background

if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(description="Train a flow matching model.")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to a checkpoint to resume training from')
    parser.add_argument('--save_dir', type=str, default='circles-eit-triangular-default-bg-x-pred-fine-tune', help='Directory to save models')
    parser.add_argument('--data_file', type=str, default='eit-circles-dtn-default-128.pt', help='Path to the dataset file')

    args = parser.parse_args()
    min_lr = 1e-9
    max_lr = 5e-6
    y_lr_mult = 1e-2
    epochs = 300
    max_steps = 400000
    batch_size = 64
    log_freq = 1000
    num_workers = 16
    save_dir = args.save_dir

    device = args.device
    dataset = torch.load(f"data/{args.data_file}", map_location="cpu")

    mesh_file = "mesh-data/mesh_128_h05.mat"
    mesh, v_h, background = get_background(mesh_file, device=device)


    dataset_X_train = dataset["train"]["dtn_map"]
    background = background.to(dataset_X_train.device)
    dataset_X_train /= background
    dataset_Y_train = dataset["train"]["media"]

    dataset_X_train = dataset_X_train.float()
    dataset_Y_train = dataset_Y_train.float()

    train_X_min = dataset_X_train.min()
    train_X_max = dataset_X_train.max()
    train_Y_min = dataset_Y_train.min()
    train_Y_max = dataset_Y_train.max()

    dataset_X_train = 2.0 * (dataset_X_train - train_X_min) / (train_X_max - train_X_min + 1e-12) - 1.0
    dataset_Y_train = 2.0 * (dataset_Y_train - train_Y_min) / (train_Y_max - train_Y_min + 1e-12) - 1.0

    if dataset_X_train.ndim == 3:
        dataset_X_train = dataset_X_train.unsqueeze(1)
    if dataset_Y_train.ndim == 3:
        dataset_Y_train = dataset_Y_train.unsqueeze(1)

    dataset_X_val = dataset["val"]["dtn_map"].float()
    dataset_X_val /= background
    dataset_Y_val = dataset["val"]["media"].float()
    dataset_X_val = 2.0 * (dataset_X_val - train_X_min) / (train_X_max - train_X_min + 1e-12) - 1.0
    dataset_Y_val = 2.0 * (dataset_Y_val - train_Y_min) / (train_Y_max - train_Y_min + 1e-12) - 1.0

    if dataset_X_val.ndim == 3:
        dataset_X_val = dataset_X_val.unsqueeze(1)
    dataset_Y_val = dataset_Y_val.unsqueeze(1)

    dataset_joint_train = torch.cat([dataset_X_train, dataset_Y_train], dim=1)
    dataset_joint_val = torch.cat([dataset_X_val, dataset_Y_val], dim=1)

    TRAIN_X_MIN = train_X_min.to(device)
    TRAIN_X_MAX = train_X_max.to(device)
    TRAIN_Y_MIN = train_Y_min.to(device)
    TRAIN_Y_MAX = train_Y_max.to(device)
    background = background.to(device)

    y_model = JiTUncond_B_8(in_channels=1, input_size=128).to(device)
    theta_model = JiT_XS_8(in_channels=1, cond_in_channels=1, out_channels=1, input_size=128).to(device)
    # if you are resuming training from a checkpoint for fine tuning itself you should just load the original weights again here, then we load new weights later
    # this is just because i wrote the constructor to need weights to load
    model = TriangularFineTune(
        y_net=y_model,
        y_net_ckpt="saved_runs/circles-eit-x-pred-dtn-JiT/checkpoints/ckp_100125.tar",
        theta_net=theta_model,
        theta_net_ckpt="saved_runs/circles-eit-cond-dtn-strong-JiT-ema-nocfg/checkpoints/best.tar",
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {n_params}")
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))

    loss_fn = get_loss_fn(model, mesh, v_h, background)

    optim = torch.optim.AdamW(
        [
            {"params": model.y_net.parameters(), "lr": max_lr * y_lr_mult},
            {"params": model.theta_net.parameters(), "lr": max_lr},
        ],
        betas=(0.9, 0.95),
        weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_steps, eta_min=min_lr)

    train = TensorDataset(dataset_joint_train.detach().clone())
    val = TensorDataset(dataset_joint_val.detach().clone())


    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    os.makedirs(f"saved_runs/{save_dir}", exist_ok=True)
    os.chdir(f"saved_runs/{save_dir}")
    
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)
    
    ckpt = args.ckpt
    best_val_loss = float("inf")
    if ckpt is not None:
        # Load checkpoint to CPU first to avoid device mismatches
        checkpoint = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        if "ema_state_dict" in checkpoint:
            ema_model.load_state_dict(checkpoint["ema_state_dict"])
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        log_step = int(checkpoint["log_step"])
        curr_epoch = int(checkpoint["epoch"])
        optim.load_state_dict(checkpoint["optim_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    else:
        log_step = 0
        curr_epoch = 0

    model = torch.compile(model)

    pbar = tqdm(range(curr_epoch, epochs + 1), desc="Epochs")
    for epoch in pbar:
        model.train()
        
        for i, (x,) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", leave=False):
            x = x.to(device)

            optim.zero_grad(set_to_none=True)

            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
                loss = loss_fn(x).float()

            if not torch.isfinite(loss):
                continue

            loss.backward()
            optim.step()
            scheduler.step()
            model_for_ema = model._orig_mod if hasattr(model, "_orig_mod") else model
            ema_model.update_parameters(model_for_ema)

            true_loss = loss.item()
            if (log_step + 1) % log_freq == 0:
                postfix = f'Step: {log_step} ({epoch}) | Loss: {true_loss:.5f}'
                pbar.set_postfix_str(postfix)
                
            log_step += 1
      
        model.eval()
        ema_model.eval()
        with torch.no_grad():
            eval_model = ema_model.module
            z0 = torch.randn(4, 2, 128, 128, device=device)

            def v_field(z, t):
                t_batch = t.expand(z.shape[0])
                t_broadcast = t_batch[:, None, None, None]
                with torch.autocast(device_type=z.device.type, dtype=torch.bfloat16):
                    x_pred = eval_model(z, t_batch)
                sigma = torch.clamp(1 - (1 - SIGMA_MIN) * t_broadcast, min=EPS)
                return x_pred - (1 - SIGMA_MIN) * (z - t_broadcast * x_pred) / sigma
            
            timesteps = torch.linspace(0.0, 1.0, steps=21).to(device)
            pred = odeint(
                func=lambda t, x: v_field(x, t),
                t = timesteps,
                y0 = z0,
                method = 'dopri5',
                atol = 1e-5,
                rtol = 1e-5, 
            )[-1]

            x_gen = pred[:, 0:1]
            y_gen = pred[:, 1:2]

            x_gen_div_bg = 0.5 * (x_gen + 1.0) * (TRAIN_X_MAX - TRAIN_X_MIN) + TRAIN_X_MIN
            y_gen_phys = 0.5 * (y_gen + 1.0) * (TRAIN_Y_MAX - TRAIN_Y_MIN) + TRAIN_Y_MIN

            x_forward_div_bg = get_dtn_from_y(
                y_gen_phys,
                v_h=v_h,
                mesh=mesh,
                background=background,
                img_size=128,
            ).unsqueeze(1)

            y_gen_phys_cpu = y_gen_phys.detach().cpu()
            x_gen_div_bg_cpu = x_gen_div_bg.detach().cpu()
            x_forward_div_bg_cpu = x_forward_div_bg.detach().cpu()

            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
            
            plt.rcParams["font.family"] = "DejaVu Serif"
            title_font = {"family": "DejaVu Serif", "weight": "bold", "size": 12}

            diff_cmap = LinearSegmentedColormap.from_list(
                "blue_black_red",
                ["#2166ac", "#000000", "#b2182b"],
                N=256,
            )

            fig, axes = plt.subplots(4, 4, figsize=(12, 12))

            diff = x_gen_div_bg_cpu[:, 0] - x_forward_div_bg_cpu[:, 0]
            v = float(diff.abs().max().item() + 1e-12)
            diff_norm = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)

            for i in range(4):
                axes[i, 0].imshow(y_gen_phys_cpu[i, 0].numpy(), cmap="Blues")
                axes[i, 0].set_title(f"Gen media {i+1}", fontdict=title_font)
                axes[i, 0].axis("off")

                axes[i, 1].imshow(x_gen_div_bg_cpu[i, 0].numpy(), cmap="magma")
                axes[i, 1].set_title("Gen DtN/bg", fontdict=title_font)
                axes[i, 1].axis("off")

                axes[i, 2].imshow(x_forward_div_bg_cpu[i, 0].numpy(), cmap="magma")
                axes[i, 2].set_title("DtN/Bg from gen media", fontdict=title_font)
                axes[i, 2].axis("off")

                axes[i, 3].imshow(diff[i].numpy(), cmap=diff_cmap, norm=diff_norm)
                axes[i, 3].set_title("Signed diff", fontdict=title_font)
                axes[i, 3].axis("off")
            
            plt.tight_layout()
            plt.savefig(f'samples/epoch_{epoch}.png', dpi=150, bbox_inches='tight')
            plt.close() 

        eval_model.eval()
        val_loss_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for (x_val,) in val_loader:
                x_val = x_val.to(device)
                t = sample_t(x_val.shape[0], device=x_val.device)
                t_broadcast = t[:, None, None, None]
                x1 = x_val.float()
                x0 = torch.randn_like(x1)
                z_t = step(t, x0, x1)
                v = target(t, x0, x1)

                with torch.autocast(device_type=x_val.device.type, dtype=torch.bfloat16):
                    x_pred = eval_model(z_t.to(dtype=x_val.dtype), t)

                sigma = torch.clamp(1 - (1 - SIGMA_MIN) * t_broadcast, min=EPS)
                v_pred = x_pred - (1 - SIGMA_MIN) * (z_t - t_broadcast * x_pred) / sigma
                batch_loss = MSELoss()(v, v_pred)
                val_loss_total += batch_loss.item()
                val_batches += 1

        val_loss = val_loss_total / max(val_batches, 1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
            checkpoint = {
                "epoch": int(epoch),
                "log_step": int(log_step),
                "best_val_loss": float(best_val_loss),
                "model_state_dict": model_to_save.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "optim_state_dict": optim.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": None,
            }
            torch.save(checkpoint, "checkpoints/best.tar")

        pbar.set_postfix_str(
            f"Step: {log_step} ({epoch}) | Val Loss: {val_loss:.5f} | Best: {best_val_loss:.5f}"
        )

    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
    checkpoint = {
        "epoch": int(epoch),
        "log_step": int(log_step),
        "best_val_loss": float(best_val_loss),
        "model_state_dict": model_to_save.state_dict(),
        "ema_state_dict": ema_model.state_dict(),
        "optim_state_dict": optim.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": None,
    }
    torch.save(checkpoint, "checkpoints/final.tar")
    with open("final_metrics.txt", "w", encoding="ascii") as f:
        f.write(f"final_val_loss: {val_loss:.6f}\n")
        f.write(f"best_saved: {best_val_loss:.6f}\n")
        f.write(f"n_params_million: {n_params / 1e6:.6f}\n")
     