# Optimized PyTorch implementation with JIT compilation
# Vectorized operations for maximum performance

import torch


class Mesh:
    def __init__(self, p, t, bdy_idx, vol_idx, device='cpu'):
        """Finite element mesh
        
        Args:
            p: (n_p, 2) array of point coordinates
            t: (n_t, 3) array of triangle node indices
            bdy_idx: (n_bdy,) array of boundary node indices
            vol_idx: (n_vol,) array of volumetric (interior) node indices
            device: torch device ('cpu' or 'cuda')
        """
        self.device = device
        self.p = torch.as_tensor(p, dtype=torch.float32, device=device)
        self.t = torch.as_tensor(t, dtype=torch.int64, device=device)
        
        self.n_t = t.shape[0]  # number of triangles
        self.n_p = p.shape[0]  # number of points
        
        self.bdy_idx = torch.as_tensor(bdy_idx, dtype=torch.int64, device=device)
        self.vol_idx = torch.as_tensor(vol_idx, dtype=torch.int64, device=device)


class V_h:
    def __init__(self, mesh):
        """Finite element space
        
        Args:
            mesh: Mesh object
        """
        self.mesh = mesh
        self.dim = mesh.n_p  # dimension of the FE space


@torch.jit.script
def _compute_all_element_stiffness(nodes_coords: torch.Tensor, sigma_vec: torch.Tensor) -> torch.Tensor:
    """Vectorized computation of element stiffness matrices
    
    Args:
        nodes_coords: (n_t, 3, 2) coordinates of triangle nodes
        sigma_vec: (n_t,) conductivity values
        
    Returns:
        S_local: (n_t, 3, 3) local stiffness matrices
    """
    n_t = nodes_coords.shape[0]
    device = nodes_coords.device
    dtype = nodes_coords.dtype
    
    # Build Pe matrix: (n_t, 3, 3) with rows=[1, x, y]
    ones = torch.ones((n_t, 3, 1), dtype=dtype, device=device)
    Pe = torch.cat([ones, nodes_coords], dim=2)  # (n_t, 3, 3)
    
    # Compute areas: |det(Pe)| / 2
    det_Pe = torch.linalg.det(Pe)  # (n_t,)
    Area = torch.abs(det_Pe) / 2.0  # (n_t,)
    
    # Compute gradients: C[1:3, :] where C = inv(Pe)
    C = torch.linalg.inv(Pe)  # (n_t, 3, 3)
    grad = C[:, 1:3, :]  # (n_t, 2, 3)
    
    # Compute local stiffness: sigma * Area * grad^T @ grad
    # grad: (n_t, 2, 3), grad^T: (n_t, 3, 2)
    # grad^T @ grad: (n_t, 3, 2) @ (n_t, 2, 3) -> (n_t, 3, 3)
    S_local = torch.einsum('e,eji,ejk->eik', sigma_vec * Area, grad, grad)
    
    return S_local


@torch.jit.script
def _compute_all_element_mass(nodes_coords: torch.Tensor) -> torch.Tensor:
    """Vectorized computation of element mass matrices
    
    Args:
        nodes_coords: (n_t, 3, 2) coordinates of triangle nodes
        
    Returns:
        M_local: (n_t, 3, 3) local mass matrices
    """
    n_t = nodes_coords.shape[0]
    device = nodes_coords.device
    dtype = nodes_coords.dtype
    
    # Build Pe matrix
    ones = torch.ones((n_t, 3, 1), dtype=dtype, device=device)
    Pe = torch.cat([ones, nodes_coords], dim=2)
    
    # Compute areas
    det_Pe = torch.linalg.det(Pe)
    Area = torch.abs(det_Pe) / 2.0  # (n_t,)
    
    # Local mass matrix template
    MK = torch.tensor([[2., 1., 1.], 
                       [1., 2., 1.],
                       [1., 1., 2.]], dtype=dtype, device=device) / 12.0
    
    # Scale by area: (n_t, 3, 3)
    M_local = Area.view(-1, 1, 1) * MK.unsqueeze(0)
    
    return M_local


@torch.jit.script
def _compute_all_partial_deriv(nodes_coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized computation of partial derivative matrices
    
    Args:
        nodes_coords: (n_t, 3, 2) coordinates of triangle nodes
        
    Returns:
        Kx_local: (n_t, 3) local x-derivative contributions
        Ky_local: (n_t, 3) local y-derivative contributions  
        Area: (n_t,) element areas
    """
    n_t = nodes_coords.shape[0]
    device = nodes_coords.device
    dtype = nodes_coords.dtype
    
    # Build Pe matrix
    ones = torch.ones((n_t, 3, 1), dtype=dtype, device=device)
    Pe = torch.cat([ones, nodes_coords], dim=2)
    
    # Compute areas
    det_Pe = torch.linalg.det(Pe)
    Area = torch.abs(det_Pe) / 2.0
    
    # Compute gradients
    C = torch.linalg.inv(Pe)
    grad = C[:, 1:3, :]  # (n_t, 2, 3)
    
    # Extract x and y gradients, scale by area
    Kx_local = grad[:, 0, :] * Area.view(-1, 1)  # (n_t, 3)
    Ky_local = grad[:, 1, :] * Area.view(-1, 1)  # (n_t, 3)
    
    return Kx_local, Ky_local, Area


def stiffness_matrix(v_h: V_h, sigma_vec: torch.Tensor) -> torch.Tensor:
    """Assemble stiffness matrix with vectorized operations
    
    Args:
        v_h: finite element space
        sigma_vec: (n_t,) conductivity values per element
        
    Returns:
        S: (n_p, n_p) stiffness matrix
    """
    t = v_h.mesh.t
    p = v_h.mesh.p
    n_t = v_h.mesh.n_t
    n_p = v_h.dim
    device = v_h.mesh.device
    dtype = v_h.mesh.p.dtype
    
    # Ensure sigma_vec is correct shape and dtype
    if not isinstance(sigma_vec, torch.Tensor):
        sigma_vec = torch.tensor(sigma_vec, dtype=dtype, device=device)
    sigma_vec = sigma_vec.reshape(-1)
    sigma_vec = sigma_vec.to(device=device, dtype=dtype)
    
    # Get node coordinates: (n_t, 3, 2)
    nodes_coords = p[t]
    
    # Compute all local stiffness matrices: (n_t, 3, 3)
    S_local = _compute_all_element_stiffness(nodes_coords, sigma_vec)
    
    # Assemble global matrix using scatter_add
    S = torch.zeros((n_p, n_p), dtype=dtype, device=device)
    
    # Create index arrays for scatter operation
    # i_idx: (n_t, 3, 3) where each [e, :, :] is nodes[e] repeated 3 times (row indices)
    # j_idx: (n_t, 3, 3) where each [e, i, :] is nodes[e] (column indices)
    nodes_i = t.unsqueeze(2).expand(-1, -1, 3)  # (n_t, 3, 3)
    nodes_j = t.unsqueeze(1).expand(-1, 3, -1)  # (n_t, 3, 3)
    
    # Flatten for scatter_add
    flat_i = nodes_i.flatten()  # (n_t * 9,)
    flat_j = nodes_j.flatten()  # (n_t * 9,)
    flat_values = S_local.flatten()  # (n_t * 9,)
    
    # Convert 2D indices to linear indices
    linear_idx = flat_i * n_p + flat_j
    
    # Scatter add using index_add (more efficient than loop)
    S_flat = S.flatten()
    S_flat.index_add_(0, linear_idx, flat_values)
    S = S_flat.view(n_p, n_p)
    
    return S


def mass_matrix(v_h: V_h) -> torch.Tensor:
    """Assemble mass matrix with vectorized operations
    
    Args:
        v_h: finite element space
        
    Returns:
        M: (n_p, n_p) mass matrix
    """
    t = v_h.mesh.t
    p = v_h.mesh.p
    n_t = v_h.mesh.n_t
    n_p = v_h.dim
    device = v_h.mesh.device
    dtype = v_h.mesh.p.dtype
    
    # Get node coordinates
    nodes_coords = p[t]
    
    # Compute all local mass matrices
    M_local = _compute_all_element_mass(nodes_coords)
    
    # Assemble global matrix
    M = torch.zeros((n_p, n_p), dtype=dtype, device=device)
    
    nodes_i = t.unsqueeze(2).expand(-1, -1, 3)
    nodes_j = t.unsqueeze(1).expand(-1, 3, -1)
    
    flat_i = nodes_i.flatten()
    flat_j = nodes_j.flatten()
    flat_values = M_local.flatten()
    
    linear_idx = flat_i * n_p + flat_j
    
    M_flat = M.flatten()
    M_flat.index_add_(0, linear_idx, flat_values)
    M = M_flat.view(n_p, n_p)
    
    return M


def projection_v_w(v_h: V_h) -> torch.Tensor:
    """Assemble projection matrix V -> W
    
    Args:
        v_h: finite element space
        
    Returns:
        P: (n_p, n_t) projection matrix
    """
    t = v_h.mesh.t
    p = v_h.mesh.p
    n_p = v_h.dim
    device = v_h.mesh.device
    dtype = v_h.mesh.p.dtype
    
    # Get node coordinates
    nodes_coords = p[t]
    n_t = t.shape[0]
    
    # Build Pe and compute areas
    ones = torch.ones((n_t, 3, 1), dtype=dtype, device=device)
    Pe = torch.cat([ones, nodes_coords], dim=2)
    det_Pe = torch.linalg.det(Pe)
    Area = torch.abs(det_Pe) / 2.0  # (n_t,)
    
    # Each node in element contributes Area/3
    M = torch.zeros((n_p, n_t), dtype=dtype, device=device)
    
    # For each element, add Area/3 to the 3 nodes
    for i in range(3):
        M.index_add_(0, t[:, i], (Area / 3.0).unsqueeze(1).expand(-1, n_t) * 
                     torch.eye(n_t, dtype=dtype, device=device))
    
    return M


def partial_deriv_matrix(v_h: V_h) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble partial derivative matrices with vectorized operations
    
    Args:
        v_h: finite element space
        
    Returns:
        Kx: (n_t, n_p) x-derivative matrix
        Ky: (n_t, n_p) y-derivative matrix
        M_w: (n_t, n_t) diagonal mass matrix in W space
    """
    t = v_h.mesh.t
    p = v_h.mesh.p
    n_p = v_h.dim
    device = v_h.mesh.device
    dtype = v_h.mesh.p.dtype
    
    # Get node coordinates
    nodes_coords = p[t]
    n_t = t.shape[0]
    
    # Compute all local contributions
    Kx_local, Ky_local, Area = _compute_all_partial_deriv(nodes_coords)
    
    # Assemble Kx and Ky
    Kx = torch.zeros((n_t, n_p), dtype=dtype, device=device)
    Ky = torch.zeros((n_t, n_p), dtype=dtype, device=device)
    
    # Scatter local contributions
    for i in range(3):
        Kx.scatter_add_(1, t[:, i:i+1], Kx_local[:, i:i+1])
        Ky.scatter_add_(1, t[:, i:i+1], Ky_local[:, i:i+1])
    
    # Diagonal mass matrix
    M_w = torch.diag(Area.to(dtype=dtype))
    
    return Kx, Ky, M_w


def dtn_map(v_h: V_h, sigma_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Dirichlet-to-Neumann map
    
    Args:
        v_h: finite element space
        sigma_vec: (n_t,) conductivity values
        
    Returns:
        DtN: (n_bdy, n_bdy) DtN map
        sol: (n_p, n_bdy) solution for each boundary basis
    """
    n_bdy_pts = len(v_h.mesh.bdy_idx)
    n_pts = v_h.mesh.p.shape[0]
    device = v_h.mesh.device
    
    vol_idx = v_h.mesh.vol_idx
    bdy_idx = v_h.mesh.bdy_idx
    
    # Build stiffness matrix
    S = stiffness_matrix(v_h, sigma_vec)
    
    # Extract submatrices
    S_ii = S[vol_idx][:, vol_idx]
    S_ib = S[vol_idx][:, bdy_idx]
    
    # Boundary data (identity)
    bdy_data = torch.eye(n_bdy_pts, device=device, dtype=S.dtype)
    
    # Right-hand side
    Fb = -S_ib @ bdy_data
    
    # Solve interior DOF
    L, info = torch.linalg.cholesky_ex(S_ii)
    if int(info.item()) != 0:
        U_vol = torch.linalg.solve(S_ii, Fb)
    else:
        U_vol = torch.cholesky_solve(Fb, L)
    
    # Assemble full solution
    sol = torch.zeros((n_pts, n_bdy_pts), device=device, dtype=S.dtype)
    sol[bdy_idx] = bdy_data
    sol[vol_idx] = U_vol
    
    # Compute flux and extract boundary values
    flux = S @ sol
    DtN = flux[bdy_idx]
    
    return DtN, sol


def adjoint(v_h: V_h, sigma_vec: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Compute adjoint solution
    
    Args:
        v_h: finite element space
        sigma_vec: (n_t,) conductivity values
        residual: (n_bdy, n_bdy) residual on boundary
        
    Returns:
        sol_adj: (n_p, n_bdy) adjoint solution
    """
    n_bdy_pts = residual.shape[0]
    n_pts = v_h.mesh.p.shape[0]
    device = v_h.mesh.device
    
    vol_idx = v_h.mesh.vol_idx
    bdy_idx = v_h.mesh.bdy_idx
    
    # Build stiffness matrix (self-adjoint operator)
    S = stiffness_matrix(v_h, sigma_vec)
    
    # Extract submatrices
    S_ii = S[vol_idx][:, vol_idx]
    S_ib = S[vol_idx][:, bdy_idx]
    
    # Right-hand side
    Fb = -S_ib @ residual
    
    # Solve interior DOF
    L, info = torch.linalg.cholesky_ex(S_ii)
    if int(info.item()) != 0:
        U_vol = torch.linalg.solve(S_ii, Fb)
    else:
        U_vol = torch.cholesky_solve(Fb, L)
    
    # Assemble full solution
    sol_adj = torch.zeros((n_pts, n_bdy_pts), device=device, dtype=S.dtype)
    sol_adj[bdy_idx] = residual
    sol_adj[vol_idx] = U_vol
    
    return sol_adj


def misfit_sigma(v_h: V_h, Data: torch.Tensor, sigma_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute misfit and gradient
    
    Args:
        v_h: finite element space
        Data: (n_bdy, n_bdy) measured data
        sigma_vec: (n_t,) conductivity values
        
    Returns:
        misfit: scalar misfit value
        grad: (n_t, 1) gradient w.r.t. sigma
    """
    # Compute forward solution
    dtn, sol = dtn_map(v_h, sigma_vec)
    
    # Compute residual
    residual = -(Data - dtn)
    
    # Compute adjoint
    sol_adj = adjoint(v_h, sigma_vec, residual)
    
    # Compute derivative matrices
    Kx, Ky, M_w = partial_deriv_matrix(v_h)
    
    # Compute derivatives
    M_w_diag_inv = torch.diag(1.0 / torch.diag(M_w))
    
    Sol_adj_x = M_w_diag_inv @ (Kx @ sol_adj)
    Sol_adj_y = M_w_diag_inv @ (Ky @ sol_adj)
    
    Sol_x = M_w_diag_inv @ (Kx @ sol)
    Sol_y = M_w_diag_inv @ (Ky @ sol)
    
    # Compute gradient
    grad = M_w @ torch.sum(Sol_adj_x * Sol_x + Sol_adj_y * Sol_y, dim=1, keepdim=True)
    
    # Compute misfit
    misfit = 0.5 * torch.sum(torch.square(residual))
    
    return misfit, grad
