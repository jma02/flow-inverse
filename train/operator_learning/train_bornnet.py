import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset

from solvers.torch_eit_fem_solver.utils import dtn_from_sigma
from solvers.torch_eit_fem_solver.fem import Mesh, V_h
import matplotlib.pyplot as plt
from torch import Tensor
from torch.nn import MSELoss

import scipy.io as sio

from models.bornnet import BornNetLinear
import argparse

torch.manual_seed(159753)
np.random.seed(159753)

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

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
    parser = argparse.ArgumentParser(description="Train a unet to parametrize an inverse operator.")
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to a checkpoint to resume training from')
    parser.add_argument('--save_dir', type=str, default='circles-eit-inverse-linear-bornnet-default', help='Directory to save models')
    parser.add_argument('--data_file', type=str, default='eit-circles-dtn-default-128.pt', help='Path to the dataset file')

    args = parser.parse_args()
    
    # Training configuration
    min_lr = 1e-8
    max_lr = 5e-4
    epochs = 400
    max_steps = 400000
    batch_size = 64
    log_freq = 1000  # sparsely log since we are training offline
    num_workers = 16
    save_dir = args.save_dir

    mesh_file = "mesh-data/mesh_128_h05.mat"
    background = get_background(mesh_file, device=args.device)

    device = args.device
    model = BornNetLinear(ch=32, in_ch=1).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=max_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_steps, eta_min=min_lr)
    # after loading the data we change working directory
    dataset = torch.load(f"data/{args.data_file}", map_location="cpu")
    

    train_data = dataset["train"]
    val_data = dataset["val"]

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

    dataset_X_val = val_data["dtn_map"].float()
    dataset_X_val /= background
    dataset_Y_val = val_data["media"].float()
    dataset_X_val = 2.0 * (dataset_X_val - train_X_min) / (train_X_max - train_X_min + 1e-12) - 1.0
    dataset_Y_val = 2.0 * (dataset_Y_val - train_Y_min) / (train_Y_max - train_Y_min + 1e-12) - 1.0
    if dataset_X_val.ndim == 3:
        dataset_X_val = dataset_X_val.unsqueeze(1)
    if dataset_Y_val.ndim == 3:
        dataset_Y_val = dataset_Y_val.unsqueeze(1)

    train = TensorDataset(dataset_X_train.detach().clone(), dataset_Y_train.detach().clone())
    test = TensorDataset(dataset_X_val.detach().clone(), dataset_Y_val.detach().clone())

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    os.makedirs(f"saved_runs/{save_dir}", exist_ok=True)
    os.chdir(f"saved_runs/{save_dir}")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)
    scaler = torch.amp.GradScaler()

    ckpt = args.ckpt
    if ckpt is not None:
        # Load checkpoint to CPU first to avoid device mismatches
        checkpoint = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        log_step = int(checkpoint["log_step"])
        curr_epoch = int(checkpoint["epoch"])
        optim.load_state_dict(checkpoint["optim_state_dict"])
        if checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
    else:
        log_step = 0
        curr_epoch = 0 

    # compile after loading checkpoints
    model = torch.compile(model)

    loss_fn = MSELoss()
    pbar = tqdm(range(curr_epoch, epochs + 1), desc="Epochs")
    for epoch in pbar:
        model.train()
        
        for i, (x,y) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", leave=False):
            x = x.to(device)
            y = y.to(device)

            optim.zero_grad(set_to_none=True)
            
            autocast_device_type = "cuda" if "cuda" in str(device) else "cpu"
            with torch.amp.autocast(device_type=autocast_device_type):
                output, intermediate = model(x)
                loss = loss_fn(output , y) + loss_fn(intermediate, y)

            scaler.scale(loss).backward()

            scaler.unscale_(optim)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optim)
            scaler.update()
            scheduler.step()

            true_loss = loss.item()
            if (log_step + 1) % log_freq == 0:
                pbar.set_postfix({'loss': f'{true_loss:.5f}', 'grad': f'{grad.item():.5f}', 'lr': f'{optim.param_groups[0]["lr"]:.3e}'})
                
            log_step += 1
        
        model.eval()
        with torch.no_grad():
            x, y = next(iter(test_loader))
            x = x[0].unsqueeze(0).to(device)
            y = y[0].unsqueeze(0).to(device)
            output, intermediate = model(x)
            
            x_vis = 0.5 * (x.squeeze() + 1.0) * (train_X_max - train_X_min) + train_X_min
            y_vis = 0.5 * (y.squeeze() + 1.0) * (train_Y_max - train_Y_min) + train_Y_min
            out_vis = 0.5 * (output.squeeze() + 1.0) * (train_Y_max - train_Y_min) + train_Y_min
            intermediate_vis = 0.5 * (intermediate.squeeze() + 1.0) * (train_Y_max - train_Y_min) + train_Y_min

            x_vis = x_vis.detach().cpu().numpy()
            y_vis = y_vis.detach().cpu().numpy()
            out_vis = out_vis.detach().cpu().numpy()
            intermediate_vis = intermediate_vis.detach().cpu().numpy()

            if x_vis.ndim == 3 and x_vis.shape[0] == 1:
                x_vis = x_vis[0]
            if y_vis.ndim == 3 and y_vis.shape[0] == 1:
                y_vis = y_vis[0]
            if out_vis.ndim == 3 and out_vis.shape[0] == 1:
                out_vis = out_vis[0]
            if intermediate_vis.ndim == 3 and intermediate_vis.shape[0] == 1:
                intermediate_vis = intermediate_vis[0]
   
            plt.rcParams["font.family"] = "DejaVu Serif"
            title_font = {"family": "DejaVu Serif", "weight": "bold", "size": 12}

            fig, axs = plt.subplots(2, 3, figsize=(14, 8))

            axs[0, 0].imshow(x_vis, cmap="magma")
            axs[0, 0].set_title("DtN", fontdict=title_font)
            axs[0, 0].axis("off")

            vmin = float(min(out_vis.min(), intermediate_vis.min(), y_vis.min()))
            vmax = float(max(out_vis.max(), intermediate_vis.max(), y_vis.max()))

            axs[0, 1].imshow(out_vis, cmap="Blues", vmin=vmin, vmax=vmax)
            axs[0, 1].set_title("Final Output", fontdict=title_font)
            axs[0, 1].axis("off")

            axs[0, 2].imshow(intermediate_vis, cmap="Blues", vmin=vmin, vmax=vmax)
            axs[0, 2].set_title("Intermediate Output", fontdict=title_font)
            axs[0, 2].axis("off")

            axs[1, 0].imshow(y_vis, cmap="Blues", vmin=vmin, vmax=vmax)
            axs[1, 0].set_title("Ground Truth", fontdict=title_font)
            axs[1, 0].axis("off")

            axs[1, 1].imshow(np.abs(out_vis - y_vis), cmap="inferno")
            axs[1, 1].set_title("Final Abs Error", fontdict=title_font)
            axs[1, 1].axis("off")

            axs[1, 2].imshow(np.abs(intermediate_vis - y_vis), cmap="inferno")
            axs[1, 2].set_title("Intermediate Abs Error", fontdict=title_font)
            axs[1, 2].axis("off")

            plt.tight_layout()
            plt.savefig(f"samples/epoch_{epoch}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        if epoch % 10 == 0 or epoch == epochs:
            model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
            checkpoint = {
                "epoch": int(epoch),
                "log_step": int(log_step),
                "model_state_dict": model_to_save.state_dict(),
                "optim_state_dict": optim.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            }
            torch.save(checkpoint, f"checkpoints/ckp_{log_step}.tar")

