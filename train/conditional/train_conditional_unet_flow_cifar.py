import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchdiffeq import odeint
from torch import Tensor
from torch.nn import MSELoss 
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Lambda, Resize, ToTensor

from models.unet import ConditionalUnet
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


def get_loss_fn(model: ConditionalUnet, *, num_classes: int, cond_drop_prob: float):
    def loss_fn(x: Tensor, y: Tensor) -> Tensor:
        cond = F.one_hot(y, num_classes=num_classes).to(dtype=x.dtype, device=x.device)
        if cond_drop_prob > 0.0:
            drop = (torch.rand(cond.shape[0], device=cond.device) < cond_drop_prob)
            if drop.any():
                cond = cond.clone()
                cond[drop] = 0.0

        t = 1.0 - torch.rand(x.shape[0], device=x.device, dtype=torch.float32) ** 2
        x_f = x.float()
        x0 = torch.randn_like(x_f)
        z_t_f = step(t, x0, x_f)
        v = target(t, x0, x_f)
        z_t = z_t_f.to(dtype=x.dtype)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            v_pred = model(z_t, t, cond)

        loss = MSELoss()(v, v_pred.float())
        return loss
    
    return loss_fn



if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(description="Train a flow matching model.")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to a checkpoint to resume training from')
    parser.add_argument('--save_dir', type=str, default='cifar10-128-flow', help='Directory to save models')
    parser.add_argument('--data_root', type=str, default='data/torchvision', help='Root directory for torchvision datasets')
    parser.add_argument('--image_size', type=int, default=128, help='Image size to train at')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')

    args = parser.parse_args()
    min_lr = 1e-8
    max_lr = 5e-4
    epochs = 125
    max_steps = 400000
    batch_size = 128
    log_freq = 1000
    num_workers = 16
    save_dir = args.save_dir

    device = args.device
    transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
        Lambda(lambda x: x * 2.0 - 1.0),
    ])
    train_set = CIFAR10(root=args.data_root, train=True, download=True, transform=transform)

    model = ConditionalUnet(cond_dim=args.num_classes, in_ch=3, out_ch=3, ch=64).to(device)
    loss_fn = get_loss_fn(model, num_classes=args.num_classes, cond_drop_prob=0.2)

    optim = torch.optim.Adam(model.parameters(), lr=max_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_steps, eta_min=min_lr)
    scaler = torch.amp.GradScaler()

    train_loader = DataLoader(
        train_set,
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
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    else:
        log_step = 0
        curr_epoch = 0

    model = torch.compile(model)
    # print number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    pbar = tqdm(range(curr_epoch, epochs + 1), desc="Epochs")
    for epoch in pbar:
        model.train()
        
        for i, (x, y) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", leave=False):
            x = x.to(device)
            y = y.to(device)

            optim.zero_grad(set_to_none=True)

            loss = loss_fn(x, y)

            if not torch.isfinite(loss):
                continue

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
            n = 4
            z0 = torch.randn(n, 3, args.image_size, args.image_size, device=device)
            class_idx = int(torch.randint(low=0, high=args.num_classes, size=(1,), device=device).item())
            labels = torch.full((n,), class_idx, device=device, dtype=torch.long)
            cond = F.one_hot(labels, num_classes=args.num_classes).float()

            def v_field(z, t):
                guidance_scale = 3.0
                t_batch = t.expand(z.shape[0])
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    v_uncond = model(z, t_batch, torch.zeros_like(cond))
                    v_cond = model(z, t_batch, cond)
                return v_uncond + guidance_scale * (v_cond - v_uncond)

            timesteps = torch.linspace(0.0, 1.0, steps=21).to(device)
            pred = odeint(
                func=lambda t, x: v_field(x, t),
                t=timesteps,
                y0=z0,
                method='dopri5',
                atol=1e-5,
                rtol=1e-5,
            )[-1]

            pred = (pred.clamp(-1.0, 1.0) + 1.0) / 2.0

            import matplotlib.pyplot as plt

            class_name = "unknown"
            if hasattr(train_set, "classes") and 0 <= class_idx < len(train_set.classes):
                class_name = train_set.classes[class_idx]

            fig, axes = plt.subplots(1, 4, figsize=(8, 2.5))
            fig.suptitle(class_name)
            for i in range(n):
                img = pred[i].permute(1, 2, 0).detach().cpu().numpy()
                axes[i].imshow(img)
                axes[i].axis("off")

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
                "scaler_state_dict": scaler.state_dict()
            }
            torch.save(checkpoint, f'checkpoints/ckp_{log_step}.tar')