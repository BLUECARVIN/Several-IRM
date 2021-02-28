import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import tqdm
from torch.autograd import grad

def pretty(vector):
    vlist = vector.view(-1).tolist()
    return "[" + ",".join("{:+.4f}".format(vi) for vi in vlist) + "]"


class InvariantRiskMinimization:
    def __init__(self, 
        lr, iteration, 
        environments, 
        verbose=True, reg=None,
        *wargs, **kwargs):
        best_reg = 0.
        best_err = 1e6

        x_val = environments[-1][0]
        y_val = environments[-1][1]

        self.lr = lr
        self.iteration = iteration
        self.verbose = verbose

        if reg:
            self.reg = reg
        else:
            self.reg = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

        for reg_val in self.reg:
            self.train(environments[:-1], reg=reg_val)
            err = (x_val @ self.solution() - y_val).pow(2).mean().item()

            if self.verbose:
                print("IRM (reg={:.3f} has {:.3f} validation error".format(reg_val, err))

            if err < best_err:
                best_err = err
                best_reg = reg_val
                best_phi = self.phi.clone()
        self.phi = best_phi

    def train(self, environments, reg=0.):
        dim_x = environments[0][0].size(1)

        self.phi = torch.nn.Parameter(torch.eye(dim_x, dim_x))
        self.w = torch.ones(dim_x, 1)
        self.w.requires_grad = True

        optimizer = torch.optim.Adam([self.phi], lr=self.lr)
        loss_func = torch.nn.MSELoss()

        for iter in tqdm.tqdm(range(self.iteration)):
            penalty = 0
            error = 0
            
            for x_e, y_e in environments:
                error_e = loss_func(x_e @ self.phi @ self.w, y_e)

                # IRMv1
                penalty += grad(error_e, self.w, create_graph=True)[0].pow(2).mean()

                error += error_e

            optimizer.zero_grad()
            (reg * error + (1 - reg) * penalty).backward()
            optimizer.step()

            if self.verbose and iter % 1000 == 0:
                w_str = pretty(self.solution())
                print("{:05d} | {:.5f} | {:.5f} | {}".format(iter, 
                reg,
                error,
                penalty,
                w_str))
    
    def solution(self):
        return (self.phi @ self.w).view(-1, 1)

        