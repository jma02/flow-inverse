import os
import numpy as np
from tqdm import tqdm

import torch
from torchdiffeq import odeint
from torch import Tensor
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from models.JiT import JiTUncond_XS_8
import argparse
import scipy.io as sio
import matplotlib.pyplot as plt
from solvers.torch_eit_fem_solver.utils import dtn_from_sigma
from solvers.torch_eit_fem_solver.fem import Mesh, V_h

torch.manual_seed(159753)
np.random.seed(159753)

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# from authors code

P_STD = 0.8
P_MEAN = -0.8
SIGMA_MIN = 1e-2

@torch.compile
def step(t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
    # broadcast t
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

def get_loss_fn(model: JiTUncond_XS_8):
    def loss_fn(x_cond: Tensor, y: Tensor) -> Tensor:
        # in their JiT code they sample t sigmoidally
        t = sample_t(y.size(0), device=y.device)
        y_f = y.float()
        x0 = torch.randn_like(y_f)
        z_t = step(t, x0, y_f)
        v = target(t, x0, y_f)

        if x_cond.ndim == 3:
            x_cond = x_cond.unsqueeze(1)

        z_t_concat = torch.cat([z_t, x_cond], dim=1)
        v_pred = model(z_t_concat, t)
        loss = MSELoss()(v, v_pred)
        return loss
    
    return loss_fn

def get_background(mesh_file, device='cuda'):
    mat_contents = sio.loadmat(mesh_file)

    p = torch.tensor(mat_contents['p'], dtype=torch.float64).to(device)
    t = torch.tensor(mat_contents['t']-1, dtype=torch.long).to(device)
    vol_idx = torch.tensor(mat_contents['vol_idx'].reshape((-1,))-1, dtype=torch.long).to(device)
    bdy_idx = torch.tensor(mat_contents['bdy_idx'].reshape((-1,))-1, dtype=torch.long).to(device)

    mesh = Mesh(p, t, bdy_idx, vol_idx)
    v_h = V_h(mesh)

    # get the dtn map
    dtn_background = dtn_from_sigma(sigma_vec=torch.ones(128, 128), v_h=v_h, mesh=mesh, img_size=128, device=device)
    
    return dtn_background 

if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(description="Train a flow matching model.")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to a checkpoint to resume training from')
    parser.add_argument('--save_dir', type=str, default='circles-eit-cond-dtn-cc-JiT-ema-nocfg', help='Directory to save models')
    parser.add_argument('--data_file', type=str, default='eit-circles-dtn-default-128.pt', help='Path to the dataset file')

    args = parser.parse_args()
    min_lr = 1e-8
    max_lr = 5e-5
    epochs = 1000
    max_steps = 400000
    batch_size = 128
    log_freq = 1000
    num_workers = 16
    save_dir = args.save_dir

    device = args.device
    dataset = torch.load(f"data/{args.data_file}", map_location="cpu")

    mesh_file = "mesh-data/mesh_128_h05.mat"
    background = get_background(mesh_file, device=args.device)


    dataset_X_train = dataset["train"]["dtn_map"].float()
    dataset_Y_train = dataset["train"]["media"].float()

    dataset_X_train = dataset_X_train / background

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
    dataset_Y_val = dataset["val"]["media"].float()

    dataset_X_val = dataset_X_val / background

    dataset_X_val = 2.0 * (dataset_X_val - train_X_min) / (train_X_max - train_X_min + 1e-12) - 1.0
    dataset_Y_val = 2.0 * (dataset_Y_val - train_Y_min) / (train_Y_max - train_Y_min + 1e-12) - 1.0

    if dataset_X_val.ndim == 3:
        dataset_X_val = dataset_X_val.unsqueeze(1)
    if dataset_Y_val.ndim == 3:
        dataset_Y_val = dataset_Y_val.unsqueeze(1)

    model = JiTUncond_XS_8(in_channels=2, out_channels=1, input_size=128).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    loss_fn = get_loss_fn(model)
    
    optim = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_steps, eta_min=min_lr)
    # after loading the data we change working directory

    train = TensorDataset(
        dataset_X_train.detach().clone(),
        dataset_Y_train.detach().clone(),
    )


    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val = TensorDataset(
        dataset_X_val.detach().clone(),
        dataset_Y_val.detach().clone(),
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
    ema_state_dict = None
    best_val_loss = float("inf")
    if ckpt is not None:
        # Load checkpoint to CPU first to avoid device mismatches
        checkpoint = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        log_step = int(checkpoint["log_step"])
        curr_epoch = int(checkpoint["epoch"])
        optim.load_state_dict(checkpoint["optim_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"]) 
        ema_state_dict = checkpoint.get("ema_state_dict")
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
    else:
        log_step = 0
        curr_epoch = 0

    model = torch.compile(model)
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    ema_model = AveragedModel(
        base_model,
        multi_avg_fn=get_ema_multi_avg_fn(0.999),
    ).to(device)
    if ema_state_dict is not None:
        ema_model.load_state_dict(ema_state_dict)

    pbar = tqdm(range(curr_epoch, epochs + 1), desc="Epochs")
    for epoch in pbar:
        model.train()
        
        for i, (x_cond, y) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", leave=False):
            x_cond = x_cond.to(device)
            y = y.to(device)

            optim.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                loss = loss_fn(x_cond, y)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            
            grad = torch.norm(
                torch.stack([p.grad.norm(2) for p in model.parameters() if p.grad is not None])
            )

            optim.step()
            ema_model.update_parameters(model._orig_mod if hasattr(model, "_orig_mod") else model)
            scheduler.step()

            true_loss = loss.item()
            if (log_step + 1) % log_freq == 0:
                pbar.set_postfix_str(f'Step: {log_step} ({epoch}) | Loss: {true_loss:.5f} | Grad: {grad.item():.5f}')
                
            log_step += 1
      
        model.eval()
        eval_model = ema_model.module
        eval_model.eval()
        with torch.no_grad():
            x_cond = dataset_X_train[0:1].to(device).repeat(4, 1, 1, 1)
            x0 = torch.randn(4, 1, 128, 128).to(device)
            def v_field(z, t):
                t_batch = t.expand(z.shape[0])
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    z_cond = torch.cat([z, x_cond], dim=1)
                    v_cond = eval_model(z_cond, t_batch)
                return v_cond
            
            timesteps = torch.linspace(0.0, 1.0, steps=20).to(device)
            pred = odeint(
                func = lambda t, x: v_field(x, t),
                t = timesteps,
                y0 = x0,
                method = 'heun2',
                atol = 1e-5,
                rtol = 1e-5, 
            )[-1]
            
            # Unnormalize predictions
            pred = 0.5 * (pred + 1.0) * (train_Y_max - train_Y_min) + train_Y_min
            
            fig, axes = plt.subplots(1, 6, figsize=(18, 3))

            x_cond_vis = 0.5 * (x_cond[0] + 1.0) * (train_X_max - train_X_min) + train_X_min
            y_gt = 0.5 * (dataset_Y_train[0:1].to(device) + 1.0) * (train_Y_max - train_Y_min) + train_Y_min

            axes[0].imshow(x_cond_vis.squeeze(0).detach().cpu().numpy(), cmap="magma")
            axes[0].set_title("Cond DtN")
            axes[0].axis("off")

            axes[1].imshow(y_gt[0].squeeze().detach().cpu().numpy(), cmap="Blues")
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            for i in range(4):
                img = pred[i].squeeze().cpu().numpy()
                axes[i + 2].imshow(img, cmap="Blues")
                axes[i + 2].set_title(f"Pred {i + 1}")
                axes[i + 2].axis("off")
            
            plt.tight_layout()
            plt.savefig(f'samples/epoch_{epoch}.png', dpi=150, bbox_inches='tight')
            plt.close() 

        eval_model.eval()
        val_loss_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for x_cond, y in val_loader:
                x_cond = x_cond.to(device)
                y = y.to(device)

                t = sample_t(y.size(0), device=y.device)
                y_f = y.float()
                x0 = torch.randn_like(y_f)
                z_t = step(t, x0, y_f)
                v = target(t, x0, y_f)

                z_t_concat = torch.cat([z_t, x_cond], dim=1)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    v_pred = eval_model(z_t_concat, t)
                batch_loss = MSELoss()(v, v_pred)

                val_loss_total += batch_loss.item()
                val_batches += 1

        val_loss = val_loss_total / max(val_batches, 1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": int(epoch),
                "log_step": int(log_step),
                "best_val_loss": float(best_val_loss),
                "model_state_dict": model._orig_mod.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "optim_state_dict": optim.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(checkpoint, 'checkpoints/best.tar')

        pbar.set_postfix_str(
            f"Step: {log_step} ({epoch}) | Val Loss: {val_loss:.5f} | Best: {best_val_loss:.5f}"
        )

    checkpoint = {
        "epoch": int(epoch),
        "log_step": int(log_step),
        "best_val_loss": float(best_val_loss),
        "model_state_dict": model._orig_mod.state_dict(),
        "ema_state_dict": ema_model.state_dict(),
        "optim_state_dict": optim.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(checkpoint, 'checkpoints/final.tar')
    with open("final_metrics.txt", "w", encoding="ascii") as f:
        f.write(f"final_val_loss: {val_loss:.6f}\n")
        f.write(f"best_saved: {best_val_loss:.6f}\n")
        f.write(f"n_params_million: {n_params / 1e6:.6f}\n")
     