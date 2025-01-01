import numpy as np

import torch
import torch.nn as nn

from cvxopt import solvers, matrix
import math

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

        # constraint
        # val = 5 * URDF's val
        # Radius of the sphere
        # Distance to the surface
        self.r = 0.05
        self.d = 0.01

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

    def constraint_valid(self, constraint_type, robot,
                         constraint_center=None,
                         point=None, normal_vector=None,
                         box_min=None, box_max=None, radius=None):
        # Assign robot state
        x, y, z = robot[0], robot[1], robot[2]

        if constraint_type == 1:
            x0, y0, z0 = constraint_center
            b = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2 - self.r ** 2
            violate = (b <= 0)
        elif constraint_type == 2:
            x0, y0, z0 = point
            a0, b0, c0 = normal_vector
            norm_score = math.sqrt(a0 ** 2 + b0 ** 2 + c0 ** 2)
            a0, b0, c0 = a0 / norm_score, b0 / norm_score, c0 / norm_score
            b = (a0 * (x - x0) + b0 * (y - y0) + c0 * (z - z0)) ** 2 - self.d ** 2
            violate = (b <= 0)
        elif constraint_type == 3:
            xmin, ymin, zmin = box_min
            xmax, ymax, zmax = box_max
            violate = (x < xmin) or (x > xmax) or (y < ymin) or (y > ymax) or (z < zmin) or (z > zmax)
        elif constraint_type == 4:
            x0, y0, z0 = constraint_center
            b = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2 - radius ** 2
            violate = (b > 0)
        return violate

    def dCBF_sphere(self, robot, u, f, g1, g2, g3, constraint_center):
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

        r = self.r

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
        h = b_safe.unsqueeze(0).to(self.device)  # [1, 1]
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

    def dCBF_surface(self, robot, u, f, g1, g2, g3, point, normal_vector):
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

        # Obstacle is a surface defined by a point on the surface and the normal vector
        x0, y0, z0 = point
        a0, b0, c0 = normal_vector
        norm_score = math.sqrt(a0**2+b0**2+c0**2)
        a0, b0, c0 = a0/norm_score, b0/norm_score, c0/norm_score

        d = self.d

        # Compute barrier function
        b = (a0 * (x - x0) + b0 * (y - y0) + c0 * (z - z0)) ** 2 - d ** 2

        Lfb = 2 * a0 * (a0 * (x - x0) + b0 * (y - y0) + c0 * (z - z0)) * f[0, 0] \
            + 2 * b0 * (a0 * (x - x0) + b0 * (y - y0) + c0 * (z - z0)) * f[0, 1] \
            + 2 * c0 * (a0 * (x - x0) + b0 * (y - y0) + c0 * (z - z0)) * f[0, 2]

        Lgb = 2 * a0 * (a0 * (x - x0) + b0 * (y - y0) + c0 * (z - z0)) * g1 \
            + 2 * b0 * (a0 * (x - x0) + b0 * (y - y0) + c0 * (z - z0)) * g2 \
            + 2 * c0 * (a0 * (x - x0) + b0 * (y - y0) + c0 * (z - z0)) * g3

        gamma = 1
        b_safe = Lfb + gamma * b
        A_safe = -Lgb

        dim = g1.shape[1]  # = 3
        G = A_safe.to(self.device)
        h = b_safe.unsqueeze(0).to(self.device)  # [1, 1]
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

    def dCBF_box(self, robot, u, f, g1, g2, g3, box_size, box_ori_inv_matrix, box_center):
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

        # box center
        x0, y0, z0 = box_center.tolist()

        # The agent need to stay in a 3D box defined by its corners
        xmin, ymin, zmin = -box_size/2
        xmax, ymax, zmax = box_size/2

        # Compute barrier function
        b1 = box_ori_inv_matrix[0, 0] * (x-x0) + box_ori_inv_matrix[0, 1] * (y-y0) + box_ori_inv_matrix[0, 2] * (z-z0)\
             - xmin
        b2 = box_ori_inv_matrix[1, 0] * (x-x0) + box_ori_inv_matrix[1, 1] * (y-y0) + box_ori_inv_matrix[1, 2] * (z-z0)\
             - ymin
        b3 = box_ori_inv_matrix[2, 0] * (x-x0) + box_ori_inv_matrix[2, 1] * (y-y0) + box_ori_inv_matrix[2, 2] * (z-z0)\
             - zmin
        b4 = xmax -\
             (box_ori_inv_matrix[0, 0] * (x-x0) + box_ori_inv_matrix[0, 1] * (y-y0) + box_ori_inv_matrix[0, 2] * (z-z0))
        b5 = ymax -\
             (box_ori_inv_matrix[1, 0] * (x-x0) + box_ori_inv_matrix[1, 1] * (y-y0) + box_ori_inv_matrix[1, 2] * (z-z0))
        b6 = zmax -\
             (box_ori_inv_matrix[2, 0] * (x-x0) + box_ori_inv_matrix[2, 1] * (y-y0) + box_ori_inv_matrix[2, 2] * (z-z0))
        b = torch.tensor([b1, b2, b3, b4, b5, b6]).unsqueeze(1).to(self.device)

        Lfb1 = box_ori_inv_matrix[0, 0] * f[0, 0] + box_ori_inv_matrix[0, 1] * f[0, 1] + box_ori_inv_matrix[0, 2] * f[0, 2]
        Lfb2 = box_ori_inv_matrix[1, 0] * f[0, 0] + box_ori_inv_matrix[1, 1] * f[0, 1] + box_ori_inv_matrix[1, 2] * f[0, 2]
        Lfb3 = box_ori_inv_matrix[2, 0] * f[0, 0] + box_ori_inv_matrix[2, 1] * f[0, 1] + box_ori_inv_matrix[2, 2] * f[0, 2]
        Lfb4 = -(box_ori_inv_matrix[0, 0] * f[0, 0] + box_ori_inv_matrix[0, 1] * f[0, 1] + box_ori_inv_matrix[0, 2] * f[0, 2])
        Lfb5 = -(box_ori_inv_matrix[1, 0] * f[0, 0] + box_ori_inv_matrix[1, 1] * f[0, 1] + box_ori_inv_matrix[1, 2] * f[0, 2])
        Lfb6 = -(box_ori_inv_matrix[2, 0] * f[0, 0] + box_ori_inv_matrix[2, 1] * f[0, 1] + box_ori_inv_matrix[2, 2] * f[0, 2])
        Lfb = torch.tensor([Lfb1, Lfb2, Lfb3, Lfb4, Lfb5, Lfb6]).unsqueeze(1).to(self.device)

        Lgb1 = box_ori_inv_matrix[0, 0] * g1 + box_ori_inv_matrix[0, 1] * g2 + box_ori_inv_matrix[0, 2] * g3
        Lgb2 = box_ori_inv_matrix[1, 0] * g1 + box_ori_inv_matrix[1, 1] * g2 + box_ori_inv_matrix[1, 2] * g3
        Lgb3 = box_ori_inv_matrix[2, 0] * g1 + box_ori_inv_matrix[2, 1] * g2 + box_ori_inv_matrix[2, 2] * g3
        Lgb4 = -(box_ori_inv_matrix[0, 0] * g1 + box_ori_inv_matrix[0, 1] * g2 + box_ori_inv_matrix[0, 2] * g3)
        Lgb5 = -(box_ori_inv_matrix[1, 0] * g1 + box_ori_inv_matrix[1, 1] * g2 + box_ori_inv_matrix[1, 2] * g3)
        Lgb6 = -(box_ori_inv_matrix[2, 0] * g1 + box_ori_inv_matrix[2, 1] * g2 + box_ori_inv_matrix[2, 2] * g3)
        Lgb = torch.cat((Lgb1, Lgb2, Lgb3, Lgb4, Lgb5, Lgb6)).to(self.device)

        gamma = 1
        b_safe = Lfb + gamma * b
        A_safe = -Lgb

        dim = g1.shape[1]  # = 3
        G = A_safe  # [6, 3]
        h = b_safe  # [6, 1]
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

    def dCBF_half_sphere(self, robot, u, f, g1, g2, g3, center, radius, current_area):
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
        x0, y0, z0 = center

        r = radius

        # Compute barrier function
        b = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2 - r ** 2

        Lfb = 2 * (x - x0) * f[0, 0] \
            + 2 * (y - y0) * f[0, 1] \
            + 2 * (z - z0) * f[0, 2]

        Lgb = 2 * (x - x0) * g1 \
            + 2 * (y - y0) * g2 \
            + 2 * (z - z0) * g3

        gamma = 1
        if current_area == 1:
            b_safe = Lfb + gamma * b
            A_safe = -Lgb
        elif current_area == 2:
            b_safe = -(Lfb + gamma * b)
            A_safe = Lgb

        dim = g1.shape[1]  # = 3
        G = A_safe.to(self.device)
        h = b_safe.unsqueeze(0).to(self.device)  # [1, 1]
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
