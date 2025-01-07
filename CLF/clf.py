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


class CLF(nn.Module):

    def __init__(self, fc_param):
        super(CLF, self).__init__()

        self.net = self.build_mlp(fc_param)
        self.x_dim = fc_param[0]
        self.u_dim = (fc_param[-1] - fc_param[0]) // fc_param[0]

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
        if self.training:
            # x.shape = [20, 1, 3], [20, 1, 6]
            net_out = self.net(x)  # [20, 1, 6], [20, 1, 18]

            # \dot{x} = f(x) + g(x) * u
            fx = net_out[:, :, :self.x_dim]  # [20, 1, 3], [20, 1, 6]
            gx = net_out[:, :, self.x_dim:]  # [20, 1, 3], [20, 1, 12]
            
            Gx = torch.reshape(gx, (x.shape[0], self.u_dim, self.x_dim)) # [20, 1, 3], [20, 2, 6]
            # print(Gx.mT.shape)    [20, 6, 2]
            
            out = fx + self.u @ Gx
            # out = fx + gx * self.u  # [20, 1, 3]

        else:
            # For test and evaluation
            # x.shape = [1, 3]
            net_out = self.net(x)  # [1, 6], [1, 18]

            # \dot{x} = f(x) + g(x) * u
            fx = net_out[:, :self.x_dim]  # [1, 3], [1, 6]
            gx = net_out[:, self.x_dim:]  # [1, 3], [1, 12]

            Gx = torch.reshape(gx, (self.u_dim, self.x_dim))
            
            # u: [1, 1], [1, 2]
            # out = fx + gx * self.u  # [1, 3]
            
            out = fx + self.u @ Gx

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

        dotV_g = 2 * (x - x0) * g[0, 0].reshape(1, 1) \
            + 2 * (y - y0) * g[0, 1].reshape(1, 1) \
            + 2 * (z - z0) * g[0, 2].reshape(1, 1)

        # dotV + epsilon * V <= 0
        # dotV_f + dotV_g * u + epsilon * V <= 0
        # dotV_g * u <= -dotV_f - epsilon * V
        # Gx <= h
        epsilon = 10
        delta = 1
        b_safe = - dotV_f - epsilon * V + delta
        A_safe = dotV_g

        dim = 1  # u.shape = [1, 1] ## g1.shape[1]  # = 3
        G = A_safe.to(self.device)
        h = b_safe.unsqueeze(0).to(self.device)  # [1, 1]?
        P = torch.eye(dim).to(self.device)  # [3, 3]
        q = -u # -u.T  # [1, 1]

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
