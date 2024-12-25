import numpy as np

import torch
import torch.nn as nn

from cvxopt import solvers, matrix

# # Load dataset
# obs = np.load('./data/obs3.npy')  # [100, 51, 19]
# obs = torch.tensor(obs).float()
# SCALING = 5.0
# acs = np.load('./data/acs3.npy')  # [100, 50 ,5]
# acs = acs * 0.01 * SCALING  # [100, 50, 3]
# acs = torch.tensor(acs).float()

# # Training data
# x_train = obs.unsqueeze(2).to(device)  # [100, 51, 1, 3]
# u_train = acs.unsqueeze(2).to(device)  # [100, 50, 1, 3]

# # Testing data
# x_test = x_train[-1, :, :, :]  # [51, 1, 3]
# u_test = u_train[-1, :, :, :]  # [50, 1, 3]

# # Initial condition for testing
# x_test0 = x_train[-1, 0, :, :]  # [1, 3]
# u_test0 = u_train[-1, 0, :, :]  # [1, 3]


def cvx_solver(P, q, G, h):
    mat_P = matrix(P.cpu().numpy())
    mat_q = matrix(q.cpu().numpy())
    mat_G = matrix(G.cpu().numpy())
    mat_h = matrix(h.cpu().numpy())

    solvers.options['show_progress'] = False

    sol = solvers.qp(mat_P, mat_q, mat_G, mat_h)

    return sol['x']


class CBF(nn.Module):

    def __init__(self, fc_param):
        super(CBF, self).__init__()

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

            net_out = self.net(x)  # [20, 1, 12]

            # \dot{x} = f(x) + g(x) * u
            fx = net_out[:, :, :3]
            gx = net_out[:, :, 3:]

            g1, g2, g3 = torch.chunk(gx, 3, dim=-1)  # [20, 1, 3]

            u = self.u  # [20, 1, 3]

            out = torch.cat([
                (g1 * u).sum(axis=2).unsqueeze(2),
                (g2 * u).sum(axis=2).unsqueeze(2),
                (g3 * u).sum(axis=2).unsqueeze(2)
            ], dim=2) + fx  # [20, 1, 3]

        else:
            # For test and evaluation

            net_out = self.net(x)  # [1, 12]

            # \dot{x} = f(x) + g(x) * u
            fx = net_out[:, :3]  # [1, 3]
            gx = net_out[:, 3:]  # [1, 9]

            g1, g2, g3 = torch.chunk(gx, 3, dim=-1)  # [1, 3]

            u = self.u  # [1, 3]

            out = torch.cat([
                (g1 * u).sum(axis=1).unsqueeze(1),
                (g2 * u).sum(axis=1).unsqueeze(1),
                (g3 * u).sum(axis=1).unsqueeze(1)
            ], dim=1) + fx  # [1, 3]

        return out

    def dCBF(self, robot, u, f, g1, g2, g3, constraint_center):
        """Enforce CBF on action

        Args:
            robot  ([1, 3]): robot state
            u  ([1, 3]): action
            f  ([1, 3]): fx
            g1 ([1, 3]): first row of gx
            g2 ([1, 3]): second row of gx
            g3 ([1, 3]): third row of gx
        """
        # Assign robot state
        x, y, z = robot[0, 0], robot[0, 1], robot[0, 2]

        # Obstacle point position
        # x0, y0, z0 = 2.66255212, -0.00543937, 3.49126458
        x0, y0, z0 = constraint_center

        # Radius
        # Radius = 5 * URDF's radius
        r = 0.05

        # Compute barrier function
        b = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2 - r ** 2

        Lfb = 2 * (x - x0) * f[0, 0] \
            + 2 * (y - y0) * f[0, 1] \
            + 2 * (z - z0) * f[0, 2]

        Lgb = 2 * (x - x0) * g1 \
            + 2 * (y - y0) * g2 \
            + 2 * (z - z0) * g3

        gamma = 1
        b_safe = Lfb + gamma * b
        A_safe = -Lgb

        dim = g1.shape[1]  # = 3
        G = A_safe.to(self.device)
        h = b_safe.unsqueeze(0).to(self.device)  # [1, 1]?
        P = torch.eye(dim).to(self.device)  # [3, 3]
        q = -u.T  # [3, 1]

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
