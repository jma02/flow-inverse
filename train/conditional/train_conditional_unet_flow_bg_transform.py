import os
import numpy as np
from tqdm import tqdm

import torch
from torchdiffeq import odeint
from torch import Tensor
from torch.nn import MSELoss 
from torch.utils.data import DataLoader, TensorDataset

from models.unet import ConditionalConcatUnet
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

SIGMA_MIN = 1e-2


@torch.compile
def step(t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
    t = t[:, None, None, None]
    mu = t * x1
    sigma = 1 - (1 - SIGMA_MIN) * t
    return sigma * x0 + mu


@torch.compile
def target(t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
    return x1 - (1 - SIGMA_MIN) * x0


def get_loss_fn(model: ConditionalConcatUnet):
    def loss_fn(x_cond: Tensor, y: Tensor) -> Tensor:
        cond_drop_prob = 0.2
        if x_cond.ndim == 3:
            x_cond = x_cond.unsqueeze(1)
        if cond_drop_prob > 0.0:
            drop = (torch.rand(x_cond.shape[0], device=x_cond.device) < cond_drop_prob)
            if drop.any():
                x_cond = x_cond.clone()
                x_cond[drop] = 0.0

        t = 1.0 - torch.rand(y.shape[0], device=y.device, dtype=torch.float32) ** 2
        y_f = y.float()
        x0 = torch.randn_like(y_f)
        z_t_f = step(t, x0, y_f)
        v = target(t, x0, y_f)
        z_t = z_t_f.to(dtype=y.dtype)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            v_pred = model(z_t, x_cond, t)

        loss = MSELoss()(v, v_pred.float())
        return loss
    
    return loss_fn

def get_background(mesh_file, device='cuda'):
    img_size = 128
    original_size = 128
    pad_size = 0

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
    parser.add_argument('--save_dir', type=str, default='circles-eit-concat-unet-flow-default-bg-transform_run1', help='Directory to save models')
    parser.add_argument('--data_file', type=str, default='eit-circles-dtn-default-128.pt', help='Path to the dataset file')

    args = parser.parse_args()
    min_lr = 1e-8
    max_lr = 5e-4
    epochs = 125
    max_steps = 400000
    batch_size = 64
    log_freq = 1000
    num_workers = 16
    save_dir = args.save_dir

    device = args.device
    dataset = torch.load(f"data/{args.data_file}", map_location="cpu")

    mesh_file = "mesh-data/mesh_128_h05.mat"
    background = get_background(mesh_file, device=args.device)


    dataset_X_train = dataset["train"]["dtn_map"]
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
    if dataset_Y_val.ndim == 3:
        dataset_Y_val = dataset_Y_val.unsqueeze(1)

    model = ConditionalConcatUnet().to(device)

    loss_fn = get_loss_fn(model)

    optim = torch.optim.Adam(model.parameters(), lr=max_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_steps, eta_min=min_lr)

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

    os.makedirs(f"saved_runs/{save_dir}", exist_ok=True)
    os.chdir(f"saved_runs/{save_dir}")
    
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)
    
    ckpt = args.ckpt
    if ckpt is not None:
        # Load checkpoint to CPU first to avoid device mismatches
        checkpoint = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
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
        
        for i, (x, y) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", leave=False):
            x = x.to(device)
            y = y.to(device)

            optim.zero_grad(set_to_none=True)

            loss = loss_fn(x, y).float()

            if not torch.isfinite(loss):
                continue

            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()

            true_loss = loss.item()
            if (log_step + 1) % log_freq == 0:
                postfix = f'Step: {log_step} ({epoch}) | Loss: {true_loss:.5f} | Grad: {grad.item():.5f}'
                pbar.set_postfix_str(postfix)
                
            log_step += 1
      
        model.eval()
        with torch.no_grad():
            x_cond = dataset_X_val[0:1].to(device).repeat(4, 1, 1, 1)

            z0 = torch.randn(4, 1, 128, 128, device=device)

            def v_from_x_pred(z, t):
                guidance_scale = 5.0
                t_batch = t.expand(z.shape[0])
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    v_uncond = model(z, torch.zeros_like(x_cond), t_batch)
                    v_cond = model(z, x_cond, t_batch)
                return v_uncond + guidance_scale * (v_cond - v_uncond)
            
            timesteps = torch.linspace(0.0, 1.0, steps=21).to(device)
            pred = odeint(
                func = lambda t, x: v_from_x_pred(x, t),
                t = timesteps,
                y0 = z0,
                method = 'dopri5',
                atol = 1e-5,
                rtol = 1e-5, 
            )[-1]
            
            # Unnormalize predictions
            pred = 0.5 * (pred + 1.0) * (train_Y_max - train_Y_min) + train_Y_min
            
            # Create 2x2 grid
            import matplotlib.pyplot as plt
            
            plt.rcParams["font.family"] = "DejaVu Serif"
            title_font = {"family": "DejaVu Serif", "weight": "bold", "size": 12}

            fig, axes = plt.subplots(1, 6, figsize=(18, 3))

            x_cond_vis = 0.5 * (x_cond[0] + 1.0) * (train_X_max - train_X_min) + train_X_min
            y_gt = 0.5 * (dataset_Y_val[0:1].to(device) + 1.0) * (train_Y_max - train_Y_min) + train_Y_min

            axes[0].imshow(x_cond_vis.squeeze(0).detach().cpu().numpy(), cmap="magma")
            axes[0].set_title("Cond DtN", fontdict=title_font)
            axes[0].axis("off")

            axes[1].imshow(y_gt[0].squeeze().detach().cpu().numpy(), cmap="Blues")
            axes[1].set_title("Ground Truth", fontdict=title_font)
            axes[1].axis("off")

            for i in range(4):
                img = pred[i].squeeze().cpu().numpy()
                axes[i + 2].imshow(img, cmap="Blues")
                axes[i + 2].set_title(f"Pred {i + 1}", fontdict=title_font)
                axes[i + 2].axis("off")
            
            plt.tight_layout()
            plt.savefig(f'samples/epoch_{epoch}.png', dpi=150, bbox_inches='tight')
            plt.close() 

        if epoch % 10 == 0 or epoch == epochs:
            model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
            checkpoint = {
                "epoch": int(epoch),
                "log_step": int(log_step),
                "model_state_dict": model_to_save.state_dict(),
                "optim_state_dict": optim.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(checkpoint, f'checkpoints/ckp_{log_step}.tar')
     