import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint

from models.unet import ConditionalConcatUnet

torch.manual_seed(159753)
np.random.seed(159753)

from solvers.torch_eit_fem_solver.utils import dtn_from_sigma
from solvers.torch_eit_fem_solver.fem import Mesh, V_h
import scipy.io as sio

# perf knobs
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

SIGMA_MIN = 1e-2

# POT
from ot import bregman, emd


# ---------------------------
# OT helpers (POT)  -- very similar to authors' style: EMD/Sinkhorn + row-normalize + multinomial
# ---------------------------

def emd_map(cost_matrix: torch.Tensor, kmax: int = 1_000_000) -> torch.Tensor:
    n0, n1 = cost_matrix.shape
    a = torch.ones(n0, device=cost_matrix.device) / n0
    b = torch.ones(n1, device=cost_matrix.device) / n1
    return emd(a=a, b=b, M=cost_matrix, numItermax=kmax)

def sinkhorn_map(cost_matrix: torch.Tensor, reg: float = 1e-1, kmax: int = 1_000) -> torch.Tensor:
    n0, n1 = cost_matrix.shape
    a = torch.ones(n0, device=cost_matrix.device) / n0
    b = torch.ones(n1, device=cost_matrix.device) / n1
    return bregman.sinkhorn_log(a=a, b=b, M=cost_matrix, reg=reg, numItermax=kmax)

def ot_matching(cost_matrix: torch.Tensor, method: str = "emd", reg: float = 1e-1) -> torch.Tensor:
    """
    Returns indices j for each i by sampling j ~ rho(j|i) from a row-normalized OT plan.
    (This is the same idea the authors use: build plan -> normalize rows -> multinomial.)
    """
    if method == "emd":
        P = emd_map(cost_matrix)
    elif method == "sinkhorn":
        P = sinkhorn_map(cost_matrix, reg=reg)
    else:
        raise ValueError(f"Unknown OT method: {method}")

    row_sums = P.sum(1, keepdim=True)
    P = torch.where(row_sums > 0, P / row_sums, torch.full_like(P, 1.0 / P.shape[1]))

    j = torch.multinomial(P, num_samples=1, replacement=True).squeeze(1)
    return j


# ---------------------------
# minibatch COT pairing (OT computed OUTSIDE compile graph)
# ---------------------------

def _pool_for_ot(Y: Tensor, pool: int) -> Tensor:
    if pool is None or pool <= 1:
        return Y
    # cheap embedding to avoid distance concentration on huge 128x128
    return F.avg_pool2d(Y, kernel_size=pool, stride=pool)

@torch._dynamo.disable
def minibatch_cot_pairing(
    y0: Tensor, u0: Tensor,
    y1: Tensor, u1: Tensor,
    eps: float = 1e-5,
    method: str = "emd",
    reg: float = 1e-1,
    y_pool: int = 1,
) -> tuple[Tensor, Tensor]:
    """
    Compute b×b cost and OT-match target batch (y1,u1) to source batch (y0,u0).
    Returns (y1_paired, u1_paired) aligned to rows of (y0,u0).

    NOTE: In the authors' Darcy code, the model is conditioned on the *target-side* y after pairing.
          We'll mirror that below (condition on y1_paired), which is typically more stable.
    """
    # flatten per-sample (NOT over all pixels)
    y0e = _pool_for_ot(y0, y_pool).flatten(start_dim=1)
    y1e = _pool_for_ot(y1, y_pool).flatten(start_dim=1)
    u0v = u0.flatten(start_dim=1)
    u1v = u1.flatten(start_dim=1)

    C_y = torch.cdist(y0e, y1e, p=2).pow(2)
    C_u = torch.cdist(u0v, u1v, p=2).pow(2)
    C = C_y + eps * C_u

    idx = ot_matching(C, method=method, reg=reg)
    return y1[idx], u1[idx]


# ---------------------------
# FM path/target (your original)
# ---------------------------

@torch.compile
def step(t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
    t = t[:, None, None, None]
    mu = t * x1
    sigma = 1 - (1 - SIGMA_MIN) * t
    return sigma * x0 + mu

@torch.compile
def target(t: Tensor, x0: Tensor, x1: Tensor) -> torch.Tensor:
    return x1 - (1 - SIGMA_MIN) * x0


# ---------------------------
# Background / data utils (your original)
# ---------------------------

def get_background(mesh_file: str, device: str = 'cuda'):
    mat_contents = sio.loadmat(mesh_file)

    p = torch.tensor(mat_contents['p'], dtype=torch.float64, device=device)
    t = torch.tensor(mat_contents['t'] - 1, dtype=torch.long, device=device)
    vol_idx = torch.tensor(mat_contents['vol_idx'].reshape((-1,)) - 1, dtype=torch.long, device=device)
    bdy_idx = torch.tensor(mat_contents['bdy_idx'].reshape((-1,)) - 1, dtype=torch.long, device=device)

    mesh = Mesh(p, t, bdy_idx, vol_idx)
    v_h = V_h(mesh)

    dtn_background = dtn_from_sigma(
        sigma_vec=torch.ones(128, 128),
        v_h=v_h,
        mesh=mesh,
        img_size=128,
        device=device,
    )
    return dtn_background


# ---------------------------
# Loss (more faithful to authors' *usage* of OT)
#   - two independent batches: source gives Y0; target gives (Y1,U1)
#   - U0 is noise in U-space
#   - OT pairs target to source using cost(Y,U)
#   - IMPORTANT: condition the net on Y1_paired (authors typically use target-side y after pairing)
#   - CFG dropout is applied to the conditioning used for the model, not the OT cost
# ---------------------------

def make_loss_two_batch(
    model: ConditionalConcatUnet,
    eps: float,
    ot_method: str,
    ot_reg: float,
    cond_drop_prob: float,
    y_pool: int,
):
    mse = MSELoss()

    def loss_two_batch(batch_src, batch_tgt, device: str) -> Tensor:
        xA, _  = batch_src      # source provides Y0 (DtN)
        xB, yB = batch_tgt      # target provides (Y1,U1) (DtN, media)

        xA = xA.to(device, non_blocking=True)
        xB = xB.to(device, non_blocking=True)
        yB = yB.to(device, non_blocking=True)

        if xA.ndim == 3: xA = xA.unsqueeze(1)
        if xB.ndim == 3: xB = xB.unsqueeze(1)
        if yB.ndim == 3: yB = yB.unsqueeze(1)

        b = xA.shape[0]

        # "Y" is DtN
        Y0 = xA.float()   # used only in OT cost (source-side marginal)
        Y1 = xB.float()   # target-side conditioning paired with U1

        # "U" is media/sigma field
        U1 = yB.float()
        U0 = torch.randn_like(U1)  # product-measure: independent noise in U-space

        # OT pairing chooses rho and reindexes the TARGET batch to align with SOURCE batch
        # returns (Y1_paired, U1_paired) aligned to rows of (Y0,U0)
        with torch.no_grad():
            Y1_paired, U1_paired = minibatch_cot_pairing(
                y0=Y0, u0=U0,
                y1=Y1, u1=U1,
                eps=eps, method=ot_method, reg=ot_reg, y_pool=y_pool
            )

        # CFG-style dropout for conditioning (do NOT apply to OT cost)
        Y_model = Y1_paired
        if cond_drop_prob > 0.0:
            drop = (torch.rand(b, device=Y_model.device) < cond_drop_prob)
            if drop.any():
                Y_model = Y_model.clone()
                Y_model[drop] = 0.0

        # sample time
        t = 1.0 - torch.rand(b, device=U1.device, dtype=torch.float32) ** 2

        z_t = step(t, U0, U1_paired).to(dtype=U1.dtype)
        v   = target(t, U0, U1_paired)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            v_pred = model(z_t, Y_model, t)

        return mse(v, v_pred.float())

    return loss_two_batch


# ---------------------------
# Main training script (tqdm style closer to your original)
# ---------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train conditional FM with minibatch COT (two-batch).")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='circles-eit-concat-unet-flow-default-bg-transform_ot_3')
    parser.add_argument('--data_file', type=str, default='eit-circles-dtn-default-128.pt')

    # OT knobs
    parser.add_argument('--ot_method', type=str, default='emd', choices=['emd', 'sinkhorn'])
    parser.add_argument('--ot_eps', type=float, default=1e-5)
    parser.add_argument('--ot_reg', type=float, default=1e-1)   # only used for sinkhorn
    parser.add_argument('--ot_y_pool', type=int, default=8, help='avgpool factor for Y in OT cost (1 = no pooling)')
    parser.add_argument('--cond_drop_prob', type=float, default=0.2)

    args = parser.parse_args()

    min_lr = 1e-8
    max_lr = 5e-4
    epochs = 125
    max_steps = 400000
    batch_size = 64
    log_freq = 1000
    num_workers = 16

    device = args.device
    save_dir = args.save_dir

    dataset = torch.load(f"data/{args.data_file}", map_location="cpu")

    mesh_file = "mesh-data/mesh_128_h05.mat"
    background = get_background(mesh_file, device=device).float()

    # --- train ---
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

    # --- val ---
    dataset_X_val = (dataset["val"]["dtn_map"].float() / background)
    dataset_Y_val = dataset["val"]["media"].float()

    dataset_X_val = 2.0 * (dataset_X_val - train_X_min) / (train_X_max - train_X_min + 1e-12) - 1.0
    dataset_Y_val = 2.0 * (dataset_Y_val - train_Y_min) / (train_Y_max - train_Y_min + 1e-12) - 1.0

    if dataset_X_val.ndim == 3:
        dataset_X_val = dataset_X_val.unsqueeze(1)
    if dataset_Y_val.ndim == 3:
        dataset_Y_val = dataset_Y_val.unsqueeze(1)

    model = ConditionalConcatUnet().to(device)

    loss_two_batch = make_loss_two_batch(
        model=model,
        eps=args.ot_eps,
        ot_method=args.ot_method,
        ot_reg=args.ot_reg,
        cond_drop_prob=args.cond_drop_prob,
        y_pool=args.ot_y_pool,
    )

    optim = torch.optim.Adam(model.parameters(), lr=max_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_steps, eta_min=min_lr)

    train_ds = TensorDataset(dataset_X_train.detach().clone(), dataset_Y_train.detach().clone())
    train_loader = DataLoader(
        train_ds,
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
        checkpoint = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        log_step = int(checkpoint["log_step"])
        curr_epoch = int(checkpoint["epoch"])
        optim.load_state_dict(checkpoint["optim_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    else:
        log_step = 0
        curr_epoch = 0

    # compile the model (OT is excluded via torch._dynamo.disable)
    model = torch.compile(model)

    # source iterator (independent stream)
    src_iter = iter(train_loader)

    pbar = tqdm(range(curr_epoch, epochs + 1), desc="Epochs")
    for epoch in pbar:
        model.train()

        # iterate over target stream with your original tqdm style
        for i, (x_tgt, y_tgt) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", leave=False):
            try:
                batch_src = next(src_iter)
            except StopIteration:
                src_iter = iter(train_loader)
                batch_src = next(src_iter)

            batch_tgt = (x_tgt, y_tgt)

            optim.zero_grad(set_to_none=True)
            loss = loss_two_batch(batch_src, batch_tgt, device=device).float()

            if not torch.isfinite(loss):
                continue

            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()

            if (log_step + 1) % log_freq == 0:
                postfix = f"Step: {log_step} ({epoch}) | Loss: {loss.item():.5f} | Grad: {grad.item():.5f}"
                pbar.set_postfix_str(postfix)

            log_step += 1

        # ---------------------------
        # quick sampling viz (your original)
        # ---------------------------
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
                func=lambda t, x: v_from_x_pred(x, t),
                t=timesteps,
                y0=z0,
                method='dopri5',
                atol=1e-5,
                rtol=1e-5,
            )[-1]

            # Unnormalize predictions
            pred = 0.5 * (pred + 1.0) * (train_Y_max - train_Y_min) + train_Y_min

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

            for k in range(4):
                img = pred[k].squeeze().cpu().numpy()
                axes[k + 2].imshow(img, cmap="Blues")
                axes[k + 2].set_title(f"Pred {k + 1}", fontdict=title_font)
                axes[k + 2].axis("off")

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
                "scaler_state_dict": None,
            }
            torch.save(checkpoint, f'checkpoints/ckp_{log_step}.tar')
