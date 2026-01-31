import os
import numpy as np
from tqdm import tqdm

import torch
from torchdiffeq import odeint
from torch import Tensor
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset

from models.unet import Unet
import argparse
import matplotlib.pyplot as plt

torch.manual_seed(159753)
np.random.seed(159753)

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
    # broadcast t
    t = t[:, None, None, None]
    mu = t * x1
    sigma = 1 - (1 - SIGMA_MIN) * t
    # interpolate
    return sigma * x0 + mu

@torch.compile
def target(t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
    # derivative
    return x1 - (1 - SIGMA_MIN) * x0

def get_loss_fn(model: Unet):
    def loss_fn(batch: Tensor) -> Tensor:
        t = torch.rand(batch.shape[0], device=batch.device)
        x0 = 3 * torch.ones_like(batch) 
        eps = torch.randn_like(batch)

        x0 = x0 + 0.5*eps

        xt = step(t, x0, batch)
        pred_vel = model(xt, t)
        true_vel = target(t, x0, batch)

        loss = MSELoss()(pred_vel, true_vel)
        return loss
    
    return loss_fn



if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(description="Train a flow matching model.")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to a checkpoint to resume training from')
    parser.add_argument('--save_dir', type=str, default='circles-eit-v-pred-threes-to-circs-no-norm', help='Directory to save models')
    parser.add_argument('--data_file', type=str, default='eit-circles-dataset-128.pt', help='Path to the dataset file')

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
    model = Unet(ch=32).to(device)

    loss_fn = get_loss_fn(model)
    
    optim = torch.optim.Adam(model.parameters(), lr=max_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_steps, eta_min=min_lr)
    # after loading the data we change working directory
    dataset = torch.load(f"data/{args.data_file}")

    dataset_train = dataset["train"]

    train = TensorDataset(dataset_train.detach().clone())


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
    scaler = torch.amp.GradScaler()

    ckpt = args.ckpt
    if ckpt is not None:
        # Load checkpoint to CPU first to avoid device mismatches
        checkpoint = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        log_step = int(checkpoint["log_step"])
        curr_epoch = int(checkpoint["epoch"])
        optim.load_state_dict(checkpoint["optim_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    else:
        log_step = 0
        curr_epoch = 0
    
    model = torch.compile(model)

    amp_device_type = "cuda" if device.startswith("cuda") else device

    pbar = tqdm(range(curr_epoch, epochs + 1), desc="Epochs")
    for epoch in pbar:
        model.train()
        
        for i, (x,) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", leave=False):
            x = x.to(device)

            optim.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type=amp_device_type):
                loss = loss_fn(x)

            scaler.scale(loss).backward()

            scaler.unscale_(optim)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            prev_scale = scaler.get_scale()
            scaler.step(optim)
            scaler.update()

            if scaler.get_scale() >= prev_scale:
                scheduler.step()

            true_loss = loss.item()
            if (log_step + 1) % log_freq == 0:
                pbar.set_postfix_str(f'Step: {log_step} ({epoch}) | Loss: {true_loss:.5f} | Grad: {grad.item():.5f}')
                
            log_step += 1
      
        model.eval()
        with torch.no_grad():
            x0 = 3 * torch.ones(4, 1, 128, 128).to(device)
            x0 = x0 + 0.5*torch.randn_like(x0)

            def v_field(z, t):
                t_batch = t.expand(z.shape[0])
                return model(z, t_batch)

            timesteps = torch.linspace(0.0, 1.0, steps=5).to(device)
            pred = odeint(
                func=lambda t, x: v_field(x, t),
                t=timesteps,
                y0=x0,
                method='dopri5',
                atol=1e-5,
                rtol=1e-5,
            )[-1]

            fig, axes = plt.subplots(1, 4, figsize=(12, 3))

            for i in range(4):
                img = pred[i].squeeze().cpu().numpy()
                axes[i].imshow(img, cmap="Blues")
                axes[i].axis("off")
            
            plt.tight_layout()
            plt.savefig(f'samples/epoch_{epoch}.png', dpi=150, bbox_inches='tight')
            plt.close()

        if epoch % 10 == 0 or epoch == epochs:
            checkpoint = {
                "epoch": int(epoch),
                "log_step": int(log_step),
                "model_state_dict": model._orig_mod.state_dict(),
                "optim_state_dict": optim.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            }
            torch.save(checkpoint, f'checkpoints/ckp_{log_step}.tar')