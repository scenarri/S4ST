import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from utils import *
import torch.nn.functional as F
import torch_dct as dct
import scipy.stats as st

class decowa_transformer():
    def __init__(self, mesh_width=3, mesh_height=3, rho=0.01,
                 num_warping=20, noise_scale=2, **kwargs):
        super().__init__()
        self.num_warping = num_warping
        self.noise_scale = noise_scale
        self.mesh_width = mesh_width
        self.mesh_height = mesh_height
        self.rho = rho
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def vwt(self, x, noise_map):
        n, c, w, h = x.size()
        X = grid_points_2d(self.mesh_width, self.mesh_height, self.device)
        Y = noisy_grid(self.mesh_width, self.mesh_height, noise_map, self.device)
        tpsb = TPS(size=(h, w), device=self.device)
        warped_grid_b = tpsb(X[None, ...], Y[None, ...])
        warped_grid_b = warped_grid_b.repeat(x.shape[0], 1, 1, 1)
        vwt_x = torch.grid_sampler_2d(x, warped_grid_b, 0, 0, False)
        return vwt_x

    def update_noise_map(self, x, target_labels, model):
        x.requires_grad = False
        noise_map = (torch.rand([self.mesh_height - 2, self.mesh_width - 2, 2]) - 0.5) * self.noise_scale
        for _ in range(1):
            noise_map.requires_grad = True
            vwt_x = self.vwt(x, noise_map)
            logits = model(vwt_x)
            loss = CE_Margin(logits, target_labels=target_labels)
            grad = torch.autograd.grad(loss, noise_map, create_graph=False, retain_graph=False)[0]
            noise_map = noise_map.detach() - self.rho * grad
        return noise_map.detach()

    def transform(self, x, target_labels, model):
        noise_map_hat = self.update_noise_map(x.clone().detach(), target_labels, model)
        vwt_x = self.vwt(x, noise_map_hat)
        return vwt_x





def K_matrix(X, Y):
    eps = 1e-9

    D2 = torch.pow(X[:, :, None, :] - Y[:, None, :, :], 2).sum(-1)
    K = D2 * torch.log(D2 + eps)
    return K

def P_matrix(X):
    n, k = X.shape[:2]
    device = X.device

    P = torch.ones(n, k, 3, device=device)
    P[:, :, 1:] = X
    return P


class TPS_coeffs(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X, Y):

        n, k = X.shape[:2]  # n = 77, k =2
        device = X.device

        Z = torch.zeros(1, k + 3, 2, device=device)
        P = torch.ones(n, k, 3, device=device)
        L = torch.zeros(n, k + 3, k + 3, device=device) # [1, 80, 80]
        K = K_matrix(X, X)

        P[:, :, 1:] = X
        Z[:, :k, :] = Y
        L[:, :k, :k] = K
        L[:, :k, k:] = P
        L[:, k:, :k] = P.permute(0, 2, 1)

        # Q = torch.solve(Z, L)[0]
        Q = torch.linalg.solve(L, Z)
        return Q[:, :k], Q[:, k:]

class TPS(torch.nn.Module):
    def __init__(self, size: tuple = (256, 256), device=None):
        super().__init__()
        h, w = size
        self.size = size
        self.device = device
        self.tps = TPS_coeffs()
        grid = torch.ones(1, h, w, 2, device=device)
        grid[:, :, :, 0] = torch.linspace(-1, 1, w)
        grid[:, :, :, 1] = torch.linspace(-1, 1, h)[..., None]
        self.grid = grid.view(-1, h * w, 2)

    def forward(self, X, Y):
        """Override abstract function."""
        h, w = self.size
        W, A = self.tps(X, Y)
        U = K_matrix(self.grid, X)
        P = P_matrix(self.grid)
        grid = P @ A + U @ W
        return grid.view(-1, h, w, 2)

def grid_points_2d(width, height, device):
    xx, yy = torch.meshgrid(
        [torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device)])
    return torch.stack([yy, xx], dim=-1).contiguous().view(-1, 2)

def noisy_grid(width, height, noise_map, device):
    """
    Make uniform grid points, and add noise except for edge points.
    """
    grid = grid_points_2d(width, height, device)
    mod = torch.zeros([height, width, 2], device=device)
    mod[1:height - 1, 1:width - 1, :] = noise_map
    return grid + mod.reshape(-1, 2)
