import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    """
    A simple MLP for colored mnist
    """
    def __init__(self, hidden_dim, grayscale=False):
        super().__init__()
        if grayscale:
            linear1 = nn.Linear(14 * 14, hidden_dim)
        else:
            linear1 = nn.Linear(2 * 14 * 14, hidden_dim)
        
        linear2 = nn.Linear(hidden_dim, hidden_dim)
        linear3 = nn.Linear(hidden_dim, 1)

        for lin in [linear1, linear2, linear3]:
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.mlp = nn.Sequential(
            linear1,
            nn.ReLU(True),
            linear2,
            nn.ReLU(True),
            linear3
        )
        self.grayscale = grayscale
    
    def forward(self, inputs):
        if self.grayscale:
            out = inputs.view(inputs.shape[0], 2, 14 * 14).sum(dim=1)
        else:
            out = inputs.view(inputs.shape[0], 2 * 14 * 14)
        out = self.mlp(out)
        return out