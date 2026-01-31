# Utility functions for PyTorch JIT version

from typing import Optional

import torch
import torch.nn.functional as F
from .fem import Mesh, V_h


def interpolate_pts_torch(
    known_vals: torch.Tensor,
    interp_pts: torch.Tensor,
    img_size: Optional[int] = None,
    device: str = 'cpu',
) -> torch.Tensor:
    """Interpolate values at given points using bilinear sampling
    
    Args:
        known_vals: (img_size, img_size) or (img_size * img_size,) values on regular grid
        interp_pts: (n_pts, 2) coordinates in [-1, 1] x [-1, 1]
        img_size: grid resolution (if omitted, inferred from known_vals)
        device: torch device
        
    Returns:
        interp_vals: (n_pts,) interpolated values
    """
    if known_vals.ndim == 2:
        if img_size is None:
            img_size = int(known_vals.shape[0])
        known_vals = known_vals.reshape(-1)
    elif known_vals.ndim == 3 and known_vals.shape[0] == 1:
        if img_size is None:
            img_size = int(known_vals.shape[1])
        known_vals = known_vals.reshape(-1)
    else:
        known_vals = known_vals.reshape(-1)

    if img_size is None:
        n = int(known_vals.numel())
        s = int(round(n ** 0.5))
        if s * s != n:
            raise ValueError(f"Cannot infer square img_size from known_vals with {n} elements")
        img_size = s

    # Reshape to 2D image
    sigma_img = known_vals.to(device).view(1, 1, img_size, img_size)
    
    # Normalize and reshape for grid_sample
    interp_pts = interp_pts.to(device)
    grid = interp_pts.clone()
    grid[:, 0] = grid[:, 0].clamp(-1, 1)
    grid[:, 1] = grid[:, 1].clamp(-1, 1)
    grid = grid.view(1, -1, 1, 2)
    
    # Bilinear interpolation
    interp_vals = F.grid_sample(
        sigma_img.float(), grid.float(),
        mode='bilinear', padding_mode='border', align_corners=True
    ).view(-1)
    
    # Mask points outside unit circle
    dist = torch.sqrt(interp_pts[:, 0] ** 2 + interp_pts[:, 1] ** 2)
    interp_vals[dist >= 1.0] = 1.0
    
    return interp_vals


def dtn_from_sigma(
    sigma_vec: torch.Tensor,
    v_h: V_h,
    mesh: Optional[Mesh] = None,
    img_size: Optional[int] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    from .fem import dtn_map

    if mesh is None:
        mesh = v_h.mesh
    if device is None:
        device = mesh.device

    centroids = torch.mean(mesh.p[mesh.t], dim=1)
    sigma_vec_grid = interpolate_pts_torch(sigma_vec, centroids, img_size, device=device).to(device)
    dtn_data, _ = dtn_map(v_h, sigma_vec_grid)
    return dtn_data


def generate_GCOORD(lx: float, ly: float, nx: int, ny: int) -> torch.Tensor:
    """Generate coordinate grid
    
    Args:
        lx, ly: domain size
        nx, ny: number of points
        
    Returns:
        GCOORD: (nx * ny, 2) coordinate array
    """
    x_coords = torch.linspace(-lx / 2, lx / 2, nx)
    y_coords = torch.linspace(-ly / 2, ly / 2, ny)
    
    xv, yv = torch.meshgrid(x_coords, y_coords, indexing='xy')
    GCOORD = torch.stack([xv.flatten(), yv.flatten()], dim=1)
    
    return GCOORD


def assemble_EL_connectivity(nel: int, nnodel: int, nex: int, nx: int, device: torch.device = None) -> torch.Tensor:
    if nnodel != 4:
        raise ValueError(f"Expected nnodel == 4 for quad elements, got {nnodel}")

    iel = torch.arange(nel, device=device, dtype=torch.long)
    row = torch.div(iel, nex, rounding_mode='floor')
    ind = iel + row

    return torch.stack([ind, ind + 1, ind + nx + 1, ind + nx], dim=1)


def central_crop(img: torch.Tensor, crop_size: int) -> torch.Tensor:
    """Central crop of 2D image
    
    Args:
        img: (h, w) image
        crop_size: output size
        
    Returns:
        cropped: (crop_size, crop_size) image
    """
    h, w = img.shape
    startx = w // 2 - (crop_size // 2)
    starty = h // 2 - (crop_size // 2)
    return img[starty:starty+crop_size, startx:startx+crop_size]
