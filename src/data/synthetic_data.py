import torch
from torch import nn


class ChainEquationModel:
    """
    SEM Data
    """
    def __init__(
        self,
        dim,
        ones=True,
        scramble=False,
        hetero=True,
        hidden=False):
        
        self.hetero = hetero
        self.hidden = hidden
        self.dim = dim // 2

        if ones:
            self.W_xy = torch.eye(self.dim)
            self.W_yz = torch.eye(self.dim)

        else:
            self.W_xy = torch.randn(self.dim, self.dim) / dim
            self.W_yz = torch.randn(self.dim, self.dim) / dim

        if scramble:
            self.scramble, _ = torch.qr(torch.randn(dim, dim))
        
        else:
            self.scramble = torch.eye(dim)

        if hidden:
            self.W_hx = torch.randn(self.dim, self.dim) / dim
            self.W_hy = torch.randn(self.dim, self.dim) / dim
            self.W_hz = torch.randn(self.dim, self.dim) / dim
        else:
            self.W_hx = torch.eye(self.dim, self.dim)
            self.W_hy = torch.zeros(self.dim, self.dim)
            self.W_hz = torch.zeros(self.dim, self.dim)

    def solution(self):
        w = torch.cat([self.W_xy.sum(1), torch.zeros(self.dim)]).view(-1, 1)
        return w, self.scramble


    def __call__(self, n, env):

        h = torch.randn(n, self.dim) * env
        x = h @ self.W_hx + torch.randn(n, self.dim) * env

        if self.hetero:
            y = x @ self.W_xy + h @ self.W_hy + torch.randn(n, self.dim) * env
            z = y @ self.W_yz + h @ self.W_hz + torch.randn(n, self.dim)

        else:
            y = x @ self.W_xy + h @ self.W_hy + torch.randn(n, self.dim)
            z = y @ self.W_yz + h @ self.W_hz + torch.randn(n, self.dim) * env

        return torch.cat([x, z], 1) @ self.scramble, y.sum(1, keepdim=True)
        