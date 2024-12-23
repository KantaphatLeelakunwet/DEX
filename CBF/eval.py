import os
import argparse
import numpy as np
import torch
from torchdiffeq import odeint
from cbf import CBF

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str,
                    choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--task', type=str,
                    choices=['NeedlePick-v0', 'NeedleRegrasp-v0'], default='NeedlePick-v0')
parser.add_argument('--data_size', type=int, default=50)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--use_dcbf', action='store_true')
parser.add_argument('--exp_id', type=int, default=0)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device(
    'cuda:' + str(args.gpu)
    if torch.cuda.is_available() else 'cpu'
)

# Load dataset
obs = np.load(f'../Data/{args.task}/obs_pos.npy')  # [100, 51, 19]
obs = torch.tensor(obs).float()
SCALING = 5.0
acs = np.load(f'../Data/{args.task}/acs_pos.npy')  # [100, 50 ,5]
acs = acs * 0.01 * SCALING  # [100, 50, 3]
acs = torch.tensor(acs).float()

# Training data
x_train = obs.unsqueeze(2).to(device)  # [100, 51, 1, 3]
u_train = acs.unsqueeze(2).to(device)  # [100, 50, 1, 3]

# Testing data
x_test = x_train[-1, :, :, :]  # [51, 1, 3]
u_test = u_train[-1, :, :, :]  # [50, 1, 3]

# Initial condition for testing
x_test0 = x_train[-1, 0, :, :]  # [1, 3]
u_test0 = u_train[-1, 0, :, :]  # [1, 3]

# Set up the dimension of the network
x_dim = x_train.shape[-1]
u_dim = u_train.shape[-1]
fc_param = [x_dim, 64, x_dim + x_dim * u_dim]

# Initialize neural ODE
func = CBF(fc_param).to(device)
func.load_state_dict(torch.load(
    f"saved_model/{args.task}/{args.exp_id}/CBF10.pth"))
func.eval()

# Set up initial state
tt = torch.tensor([0., 0.1]).to(device)
x0 = x_test0.clone().detach().to(device)
pred_x = x0.unsqueeze(0)

safety = []
trajectory = []
trajectory.append(
    np.array(
        [x0[0, 0].cpu().item(), x0[0, 1].cpu().item(), x0[0, 2].cpu().item()]
    )
)

with torch.no_grad():

    for i in range(args.data_size):
        # Setup u for forward()
        # u_test[i,:,:] is the actions of test trajectory that at time i
        u_test_i = u_test[i, :, :]  # [1, 3]

        if args.use_dcbf:

            net_out = func.net(x0)  # [1, 12]

            # \dot{x} = f(x) + g(x) * u
            fx = net_out[:, :3]  # [1, 3]
            gx = net_out[:, 3:]  # [1, 9]

            g1, g2, g3 = torch.chunk(gx, 3, dim=-1)  # [1, 3]

            func.u = func.dCBF(x0, u_test_i, fx, g1, g2, g3)
        else:
            func.u = u_test_i

        pred = odeint(func, x0, tt)
        pred_x = torch.cat(
            [pred_x, pred[-1, :, :].unsqueeze(0)], dim=0)
        x0 = pred[-1, :, :]
        trajectory.append(
            np.array(
                [x0[0, 0].cpu().item(), x0[0, 1].cpu().item(), x0[0, 2].cpu().item()]
            )
        )

        # Compute the distance between robot and obstacle point
        barrier = (x0[0, 0] - 2.67054296) ** 2 \
            + (x0[0, 1] - (-0.04659402)) ** 2  \
            + (x0[0, 2] - 3.4671967) ** 2     \
            - 0.05 ** 2
        safety.append(barrier)


print("====== Safety ======")
print(torch.sum(torch.tensor(safety) < 0))
print(np.array(trajectory))
