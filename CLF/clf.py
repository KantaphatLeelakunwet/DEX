import numpy as np

import torch
import torch.nn as nn

from cvxopt import solvers, matrix


def cvx_solver(P, q, G, h):
    mat_P = matrix(P.cpu().numpy())
    mat_q = matrix(q.cpu().numpy())
    mat_G = matrix(G.cpu().numpy())
    mat_h = matrix(h.cpu().numpy())

    solvers.options['show_progress'] = False

    sol = solvers.qp(mat_P, mat_q, mat_G, mat_h)

    return sol['x']


class ODEFunc(nn.Module):

    def __init__(self, fc_param):
        super(ODEFunc, self).__init__()

        self.net = self.build_mlp(fc_param)

        # Initializing weights
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

        self.u = None
        self.device = torch.device(
            'cuda:' + str(0)
            if torch.cuda.is_available() else 'cpu'
        )

    def forward(self, t, x):
        # x.shape = [20, 1, 3]
        if self.training:

            net_out = self.net(x)  # [20, 1, 6]

            # \dot{x} = f(x) + g(x) * u
            fx = net_out[:, :, :3]  # [20, 1, 3]
            gx = net_out[:, :, 3:]  # [20, 1, 3]

            u = self.u  # [20, 1, 1]

            out = fx + gx * u  # [20, 1, 3]

        else:
            # For test and evaluation

            net_out = self.net(x)  # [1, 6]

            # \dot{x} = f(x) + g(x) * u
            fx = net_out[:, :3]  # [1, 3]
            gx = net_out[:, 3:]  # [1, 3]

            u = self.u  # [1, 1]

            out = fx + gx * u  # [1, 3]

        return out

    def dCLF(self, robot, gt, u, f, g):
        """
        Enforce CLF on action

        Args:
            robot  ([1, 3]): robot orientation
            gt ([1, 3]): orientation from expert trajectory
            u  ([1, 1]): action
            f  ([1, 3]): fx
            g  ([1, 3]): gx
        """

        # Assign robot orientation
        x, y, z = robot[0, 0], robot[0, 1], robot[0, 2]

        # Expert Demonstration's Orientation
        x0, y0, z0 = gt[0, 0], gt[0, 1], gt[0, 2]

        # Compute CLF
        V = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2

        dotV_f = 2 * (x - x0) * f[0, 0] \
            + 2 * (y - y0) * f[0, 1] \
            + 2 * (z - z0) * f[0, 2]

        dotV_g = 2 * (x - x0) * g[0, 0] \
            + 2 * (y - y0) * g[0, 1] \
            + 2 * (z - z0) * g[0, 2]

        # dotV + epsilon * V <= 0
        # dotV_f + dotV_g * u + epsilon * V <= 0
        # dotV_g * u <= -dotV_f - epsilon * V
        # Gx <= h
        epsilon = 1
        b_safe = - dotV_f - epsilon * V
        A_safe = dotV_g

        dim = 1  # u.shape = [1, 1] ## g1.shape[1]  # = 3
        G = A_safe.to(self.device)
        h = b_safe.unsqueeze(0).to(self.device)  # [1, 1]?
        P = torch.eye(dim).to(self.device)  # [3, 3]
        q = -u.T  # [1, 1]

        # NOTE: different x from above now
        x = cvx_solver(P.double(), q.double(), G.double(), h.double())

        out = []
        for i in range(dim):
            out.append(x[i])
        out = np.array(out)
        out = torch.tensor(out).float().to(self.device)
        out = out.unsqueeze(0)
        return out

    def build_mlp(self, filters, no_act_last_layer=True, activation='gelu'):
        if activation == 'gelu':
            activation = nn.GELU()
        elif activation == 'silu':
            activation = nn.SiLU()
        elif activation == 'tanh':
            activation = nn.Tanh()
        else:
            raise NotImplementedError(
                f'Not supported activation function {activation}')
        modules = nn.ModuleList()
        for i in range(len(filters)-1):
            modules.append(nn.Linear(filters[i], filters[i+1]))
            if not (no_act_last_layer and i == len(filters)-2):
                modules.append(activation)

        modules = nn.Sequential(*modules)
        return modules
