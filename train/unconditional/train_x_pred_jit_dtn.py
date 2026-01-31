import os
import numpy as np
from tqdm import tqdm

import torch
from torchdiffeq import odeint
from torch import Tensor
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from models.JiT import JiTUncond_B_8
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
@torch.compile
def interpolate_xt(t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
    # broadcast t
    t = t[:, None, None, None]
    # interpolate
    return t*x1 + (1 - t)*x0

@torch.compile
def sample_t(n: int, device=None):
    z = torch.randn(n, device=device) * P_STD + P_MEAN
    return torch.sigmoid(z)

def get_loss_fn(model: JiTUncond_B_8):
    def loss_fn(batch: Tensor) -> Tensor:
        # in their JiT code they sample t sigmoidally
        t = sample_t(batch.size(0), device=batch.device)
        t_broadcast = t[:, None, None, None]
        x0 = torch.randn_like(batch)
        z_t = interpolate_xt(t, x0, batch)

        v = (batch-z_t)/torch.clip((1-t_broadcast), min=5e-2)

        x_pred = model(z_t, t)
        v_pred = (x_pred - z_t) / torch.clip((1 - t_broadcast), min=5e-2)
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
    parser.add_argument('--save_dir', type=str, default='circles-eit-x-pred-dtn-JiT', help='Directory to save models')
    parser.add_argument('--data_file', type=str, default='eit-circles-dtn-default-128.pt', help='Path to the dataset file')

    args = parser.parse_args()
    min_lr = 1e-8
    max_lr = 5e-5
    epochs = 1000
    max_steps = 400000
    batch_size = 256
    log_freq = 1000
    num_workers = 16
    save_dir = args.save_dir

    device = args.device
    dataset = torch.load(f"data/{args.data_file}", map_location="cpu")

    mesh_file = "mesh-data/mesh_128_h05.mat"
    background = get_background(mesh_file, device=args.device)


    dataset_train = dataset["train"]["dtn_map"]
    dataset_train /= background

    train_min = dataset_train.min()
    train_max = dataset_train.max()

    dataset_train = 2.0 * (dataset_train - train_min) / (train_max - train_min + 1e-12) - 1.0

    if dataset_train.ndim == 3:
        dataset_train = dataset_train.unsqueeze(1)

    model = JiTUncond_B_8(in_channels=1, input_size=128).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    loss_fn = get_loss_fn(model)
    
    optim = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_steps, eta_min=min_lr)
    # after loading the data we change working directory

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
    
    ckpt = args.ckpt
    ema_state_dict = None
    if ckpt is not None:
        # Load checkpoint to CPU first to avoid device mismatches
        checkpoint = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        log_step = int(checkpoint["log_step"])
        curr_epoch = int(checkpoint["epoch"])
        optim.load_state_dict(checkpoint["optim_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"]) 
        ema_state_dict = checkpoint.get("ema_state_dict")
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
        
        for i, (x,) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", leave=False):
            x = x.to(device)

            optim.zero_grad(set_to_none=True)
            
            # try in fp32
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                loss = loss_fn(x)

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
        ema_model.module.eval()
        with torch.no_grad():
            x0 = torch.randn(4, 1, 128, 128).to(device)
            def v_from_x_pred(z, t):
                t_batch = t.expand(z.shape[0])
                eval_model = ema_model.module
                return (eval_model(z, t_batch) - z) / torch.clip((1 - t_batch)[:, None, None, None], min=5e-2)
            
            timesteps = torch.linspace(0.0, 1.0, steps=10).to(device)
            pred = odeint(
                func = lambda t, x: v_from_x_pred(x, t),
                t = timesteps,
                y0 = x0,
                method = 'heun2',
                atol = 1e-5,
                rtol = 1e-5, 
            )[-1]
            
            # Unnormalize predictions
            pred = 0.5 * (pred + 1.0) * (train_max - train_min) + train_min
            
            fig, axes = plt.subplots(1, 4, figsize=(12, 3))

            for i in range(4):
                img = pred[i].squeeze().cpu().numpy()
                axes[i].imshow(img, cmap="Blues")
                axes[i].axis("off")
            
            plt.tight_layout()
            plt.savefig(f'samples/epoch_{epoch}.png', dpi=150, bbox_inches='tight')
            plt.close() 

        if epoch % 100 == 0 or epoch == epochs:
            checkpoint = {
                "epoch": int(epoch),
                "log_step": int(log_step),
                "model_state_dict": model._orig_mod.state_dict(),
                "optim_state_dict": optim.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "ema_state_dict": ema_model.state_dict()
            }
            torch.save(checkpoint, f'checkpoints/ckp_{log_step}.tar')
     