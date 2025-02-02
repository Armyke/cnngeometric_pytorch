from __future__ import print_function, division
import numpy as np

import torch
import torch.nn as nn

from torch.autograd import Variable

from geotnf.point_tnf import PointTnf


class TransformedGridLoss(nn.Module):
    def __init__(self, geometric_model='affine', use_cuda=True, grid_size=20):
        super(TransformedGridLoss, self).__init__()
        self.geometric_model = geometric_model
        # define virtual grid of points to be transformed
        axis_coords = np.linspace(-1, 1, grid_size)
        self.N = grid_size * grid_size
        X, Y = np.meshgrid(axis_coords, axis_coords)
        X = np.reshape(X, (1, 1, self.N))
        Y = np.reshape(Y, (1, 1, self.N))
        P = np.concatenate((X, Y), 1)
        self.P = Variable(torch.FloatTensor(P), requires_grad=False)
        self.pointTnf = PointTnf(use_cuda)
        if use_cuda:
            self.P = self.P.cuda();

    def forward(self, theta, theta_GT):
        # expand grid according to batch size
        batch_size = theta.size()[0]
        P = self.P.expand(batch_size, 2, self.N)
        # compute transformed grid points using estimated and GT tnfs
        if self.geometric_model == 'affine':
            P_prime = self.pointTnf.affPointTnf(theta, P)
            P_prime_GT = self.pointTnf.affPointTnf(theta_GT, P)
        elif self.geometric_model == 'tps':
            P_prime = self.pointTnf.tpsPointTnf(theta.unsqueeze(2).unsqueeze(3), P)
            P_prime_GT = self.pointTnf.tpsPointTnf(theta_GT, P)
        # compute MSE loss on transformed grid points
        loss = torch.sum(torch.pow(P_prime - P_prime_GT, 2), 1)
        loss = torch.mean(loss)
        return loss


class GridLossWithMSE(nn.Module):
    def __init__(self, geometric_model='affine',
                 use_cuda=True, grid_size=20,
                 alpha=0.5):
        super(GridLossWithMSE, self).__init__()
        self.alpha = alpha
        self.geometric_model = geometric_model
        # define virtual grid of points to be transformed
        axis_coords = np.linspace(-1, 1, grid_size)
        self.N = grid_size * grid_size
        X, Y = np.meshgrid(axis_coords, axis_coords)
        X = np.reshape(X, (1, 1, self.N))
        Y = np.reshape(Y, (1, 1, self.N))
        P = np.concatenate((X, Y), 1)
        self.P = Variable(torch.FloatTensor(P), requires_grad=False)
        self.pointTnf = PointTnf(use_cuda)
        if use_cuda:
            self.P = self.P.cuda();

    def forward(self, theta, theta_GT, tb_writer=None, step=None):
        # expand grid according to batch size
        batch_size = theta.size()[0]
        P = self.P.expand(batch_size, 2, self.N)
        # compute transformed grid points using estimated and GT tnfs
        if self.geometric_model == 'affine':
            P_prime = self.pointTnf.affPointTnf(theta, P)
            P_prime_GT = self.pointTnf.affPointTnf(theta_GT, P)
        elif self.geometric_model == 'tps':
            P_prime = self.pointTnf.tpsPointTnf(theta.unsqueeze(2).unsqueeze(3), P)
            P_prime_GT = self.pointTnf.tpsPointTnf(theta_GT, P)
        # compute MSE loss on transformed grid points
        grid_loss = torch.sum(torch.pow(P_prime - P_prime_GT, 2), 1)
        grid_loss = torch.mean(grid_loss)

        # compute MSE on affinity matrices
        mse_loss = ((theta.view([-1, 2, 3]) - theta_GT)**2).mean()

        if tb_writer is not None and step is not None:
            tb_writer.add_scalar('grid loss',
                                 grid_loss.data.item(),
                                 step)
            tb_writer.add_scalar('MSE loss',
                                 mse_loss.data.item(),
                                 step)

        return self.alpha * grid_loss + (1 - self.alpha) * mse_loss

