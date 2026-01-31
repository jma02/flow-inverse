# Optimized PyTorch EIT solver with JIT compilation

import torch
from .fem import stiffness_matrix, partial_deriv_matrix, mass_matrix


class EIT:
    """Electrical Impedance Tomography solver with JIT-optimized operations"""
    
    def __init__(self, v_h):
        """Initialize EIT solver
        
        Args:
            v_h: finite element space
        """
        self.v_h = v_h
        self.build_matrices()
    
    def update_matrices(self, sigma_vec):
        """Update stiffness matrices for given conductivity
        
        Args:
            sigma_vec: (n_t,) conductivity values
        """
        vol_idx = self.v_h.mesh.vol_idx
        bdy_idx = self.v_h.mesh.bdy_idx
        
        S = stiffness_matrix(self.v_h, sigma_vec)
        self.S = S
        self.S_ii = S[vol_idx][:, vol_idx]
        self.S_ib = S[vol_idx][:, bdy_idx]
    
    def build_matrices(self):
        """Build derivative and mass matrices (conductivity-independent)"""
        self.Mass = mass_matrix(self.v_h)
        Kx, Ky, M_w = partial_deriv_matrix(self.v_h)
        
        # Compute inverse of diagonal mass matrix
        M_w_diag_inv = torch.diag(1.0 / torch.diag(M_w))
        
        self.Dx = M_w_diag_inv @ Kx
        self.Dy = M_w_diag_inv @ Ky
        self.M_w = M_w
    
    def dtn_map(self, sigma_vec):
        """Compute Dirichlet-to-Neumann map
        
        Args:
            sigma_vec: (n_t,) conductivity values
            
        Returns:
            DtN: (n_bdy, n_bdy) DtN map
            sol: (n_p, n_bdy) solution for each boundary basis
        """
        self.update_matrices(sigma_vec)
        
        n_bdy_pts = len(self.v_h.mesh.bdy_idx)
        n_pts = self.v_h.mesh.p.shape[0]
        device = self.v_h.mesh.device
        
        vol_idx = self.v_h.mesh.vol_idx
        bdy_idx = self.v_h.mesh.bdy_idx
        
        # Boundary data (identity)
        bdy_data = torch.eye(n_bdy_pts, device=device, dtype=self.S_ii.dtype)
        
        # Right-hand side
        Fb = -self.S_ib @ bdy_data
        
        # Solve interior DOF
        U_vol = torch.linalg.solve(self.S_ii, Fb)
        
        # Assemble full solution
        sol = torch.zeros((n_pts, n_bdy_pts), device=device, dtype=self.S_ii.dtype)
        sol[bdy_idx] = bdy_data
        sol[vol_idx] = U_vol
        
        # Compute flux
        flux = self.S @ sol
        
        # Extract boundary flux
        DtN = flux[bdy_idx]
        
        return DtN, sol
    
    def adjoint(self, sigma_vec, residual):
        """Compute adjoint solution
        
        Args:
            sigma_vec: (n_t,) conductivity values (unused but kept for interface)
            residual: (n_bdy, n_bdy) residual on boundary
            
        Returns:
            sol_adj: (n_p, n_bdy) adjoint solution
        """
        n_bdy_pts = len(self.v_h.mesh.bdy_idx)
        n_pts = self.v_h.mesh.p.shape[0]
        device = self.v_h.mesh.device
        
        vol_idx = self.v_h.mesh.vol_idx
        bdy_idx = self.v_h.mesh.bdy_idx
        
        # Boundary data from residual
        bdy_data = residual
        
        # Right-hand side
        Fb = -self.S_ib @ bdy_data
        
        # Solve interior DOF
        U_vol = torch.linalg.solve(self.S_ii, Fb)
        
        # Assemble full solution
        sol_adj = torch.zeros((n_pts, n_bdy_pts), device=device, dtype=self.S_ii.dtype)
        sol_adj[bdy_idx] = bdy_data
        sol_adj[vol_idx] = U_vol
        
        return sol_adj
    
    def misfit(self, Data, sigma_vec):
        """Compute misfit and gradient
        
        Args:
            Data: (n_bdy, n_bdy) measured data
            sigma_vec: (n_t,) conductivity values
            
        Returns:
            misfit: scalar misfit value (RMSE)
            grad: (n_t, 1) gradient w.r.t. sigma
        """
        # Compute forward solution
        dtn, sol = self.dtn_map(sigma_vec)
        
        # Compute residual
        residual = -(Data - dtn)
        
        # Compute adjoint
        sol_adj = self.adjoint(sigma_vec, residual)
        
        # Compute derivatives using precomputed matrices
        Sol_adj_x = self.Dx @ sol_adj
        Sol_adj_y = self.Dy @ sol_adj
        
        Sol_x = self.Dx @ sol
        Sol_y = self.Dy @ sol
        
        # Compute gradient
        grad = self.M_w @ torch.sum(Sol_adj_x * Sol_x + Sol_adj_y * Sol_y, dim=1, keepdim=True)
        
        # Compute misfit (RMSE)
        misfit = torch.sqrt(torch.sum(torch.square(residual)))
        
        return misfit, grad


# Wrapper function for torch.compile compatibility
def dtn_map_wrapper(v_h, sigma_vec):
    """Standalone DtN map function (more JIT-friendly)
    
    Args:
        v_h: finite element space
        sigma_vec: (n_t,) conductivity values
        
    Returns:
        DtN: (n_bdy, n_bdy) DtN map
    """
    from .fem import dtn_map
    DtN, _ = dtn_map(v_h, sigma_vec)
    return DtN
