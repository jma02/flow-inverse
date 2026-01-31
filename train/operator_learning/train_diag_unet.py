import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import cmocean
from torch import Tensor
from torch.nn import MSELoss

from models.unet import UnetNoTimeEmbedDiag
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


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(description="Train a unet to parametrize an inverse operator.")
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to a checkpoint to resume training from')
    parser.add_argument('--save_dir', type=str, default='circles-eit-default-unet-diag_run1', help='Directory to save models')
    parser.add_argument('--data_file', type=str, default='eit-circles-dtn-default-128.pt', help='Path to the dataset file')

    args = parser.parse_args()
    
    # Training configuration
    lr = 1e-2
    epochs = 125
    max_steps = 400000
    batch_size = 64
    log_freq = 1000  # sparsely log since we are training offline
    num_workers = 16
    save_dir = args.save_dir

    device = args.device
    model = UnetNoTimeEmbedDiag(diag_dim=128).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # after loading the data we change working directory
    dataset = torch.load(f"data/{args.data_file}", map_location="cpu")
    

    train_data = dataset["train"]
    val_data = dataset["val"]

    dataset_X_train = dataset["train"]["dtn_map"]
    dataset_Y_train = dataset["train"]["media"]

    dataset_X_train = dataset_X_train.float()
    # take the diagonal of the matrix
    dataset_X_train = torch.diagonal(dataset_X_train, dim1=1, dim2=2)
    dataset_Y_train = dataset_Y_train.float()

    if dataset_X_train.ndim == 3:
        dataset_X_train = dataset_X_train.unsqueeze(1)
    if dataset_Y_train.ndim == 3:
        dataset_Y_train = dataset_Y_train.unsqueeze(1)

    dataset_X_val = val_data["dtn_map"].float()
    dataset_X_val = torch.diagonal(dataset_X_val, dim1=1, dim2=2)
    dataset_Y_val = val_data["media"].float()
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
        model = model.to(device)
        log_step = int(checkpoint["log_step"])
        curr_epoch = int(checkpoint["epoch"])
        optim.load_state_dict(checkpoint["optim_state_dict"])
        for state in optim.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
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
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(x)
                # view as image
                output = output.view(-1, 1, 128, 128)
                loss = loss_fn(output ,y)

            scaler.scale(loss).backward()

            scaler.unscale_(optim)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optim)
            scaler.update()

            true_loss = loss.item()
            if (log_step + 1) % log_freq == 0:
                pbar.set_postfix({'loss': f'{true_loss:.5f}', 'grad': f'{grad.item():.5f}', 'lr': f'{optim.param_groups[0]["lr"]:.3e}'})
                
            log_step += 1
        
        model.eval()
        with torch.no_grad():
            x, y = next(iter(test_loader))
            x = x[0].unsqueeze(0).to(device)
            y = y[0].unsqueeze(0).to(device)
            output = model(x)
            # view as image
            output = output.view(-1, 1, 128, 128)
            x_vis = x.squeeze()
            y_vis = y.squeeze()
            out_vis = output.squeeze()

            x_vis = x_vis.detach().cpu().numpy()
            y_vis = y_vis.detach().cpu().numpy()
            out_vis = out_vis.detach().cpu().numpy()

            if x_vis.ndim == 3 and x_vis.shape[0] == 1:
                x_vis = x_vis[0]
            if x_vis.ndim == 1:
                x_vis = x_vis[None, :]
            if y_vis.ndim == 3 and y_vis.shape[0] == 1:
                y_vis = y_vis[0]
            if out_vis.ndim == 3 and out_vis.shape[0] == 1:
                out_vis = out_vis[0]
   
            plt.rcParams["font.family"] = "DejaVu Serif"
            title_font = {"family": "DejaVu Serif", "weight": "bold", "size": 12}

            fig, axs = plt.subplots(2, 2, figsize=(10, 8))

            axs[0, 0].imshow(x_vis, cmap="magma")
            axs[0, 0].set_title("DtN", fontdict=title_font)
            axs[0, 0].axis("off")

            vmin = float(min(out_vis.min(), y_vis.min()))
            vmax = float(max(out_vis.max(), y_vis.max()))

            axs[0, 1].imshow(out_vis, cmap="Blues", vmin=vmin, vmax=vmax)
            axs[0, 1].set_title("UNet Output", fontdict=title_font)
            axs[0, 1].axis("off")

            axs[1, 0].imshow(y_vis, cmap="Blues", vmin=vmin, vmax=vmax)
            axs[1, 0].set_title("Ground Truth", fontdict=title_font)
            axs[1, 0].axis("off")

            axs[1, 1].imshow(np.abs(out_vis - y_vis), cmap="inferno")
            axs[1, 1].set_title("Abs Error", fontdict=title_font)
            axs[1, 1].axis("off")

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
                "scaler_state_dict": scaler.state_dict(),
            }
            torch.save(checkpoint, f"checkpoints/ckp_{log_step}.tar")

