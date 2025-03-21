import os
import argparse
import numpy as np
import torch
from torchdiffeq import odeint
from clf import CLF

tasks = ['NeedlePick-v0', 'GauzeRetrieve-v0',
         'NeedleReach-v0', 'PegTransfer-v0', 'NeedleRegrasp-v0',]

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str,
                    choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--task', type=str, choices=tasks, default='NeedlePick-v0')
parser.add_argument('--data_size', type=int, default=50)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--use_dclf', action='store_true')
parser.add_argument('--exp_id', type=int, default=0)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device(
    'cuda:' + str(args.gpu)
    if torch.cuda.is_available() else 'cpu'
)

# Load dataset
obs = np.load(f'../Data/{args.task}/obs_orn.npy')  # [100, 51, 4]
if args.task != 'NeedleRegrasp-v0':
    obs = obs[:, :, 0:3]  # [100, 51, 3]
obs = torch.tensor(obs).float()
    
acs = np.load(f'../Data/{args.task}/acs_orn.npy')  # [100, 50 ,2]
if args.task == 'NeedleRegrasp-v0':
    acs *= np.deg2rad(15)
else:
    acs = acs[:, :, 0] * np.deg2rad(30)  # [100, 50, 1]
    acs = acs.reshape((100, 50, 1))
acs = torch.tensor(acs).float()

# Training data
x_train = obs.unsqueeze(2).to(device)  # [100, 51, 1, 3]
u_train = acs.unsqueeze(2).to(device)  # [100, 50, 1, 3]

# Testing data
x_test = x_train[-1, :, :, :]  # [51, 1, 3]
u_test = u_train[-1, :, :, :]  # [50, 1, 3]

print("Timestep 24 Expert data")
print("x:", x_test[24])
print("u:", u_test[24])
print("------------")

# Initial condition for testing
x_test0 = x_train[-1, 0, :, :]  # [1, 3]
u_test0 = u_train[-1, 0, :, :]  # [1, 3]

# Set up the dimension of the network
x_dim = x_train.shape[-1]
u_dim = u_train.shape[-1]
fc_param = [x_dim, 64, x_dim + x_dim * u_dim]

# Initialize neural ODE
func = CLF(fc_param).to(device)
func.load_state_dict(torch.load(
    f"saved_model/{args.task}/{args.exp_id}/CLF10.pth"))
func.eval()

# Set up initial state
tt = torch.tensor([0., 0.1]).to(device)
x0 = x_test0.clone().detach().to(device)
pred_x = x0.unsqueeze(0)

total_loss = []


with torch.no_grad():

    for i in range(args.data_size):
        # Setup u for forward()
        # u_test[i,:,:] is the actions of test trajectory that at time i
        u_test_i = u_test[i, :, :]  # [1, 1]

        if args.use_dclf:

            net_out = func.net(x0)  # [1, 6]

            # \dot{x} = f(x) + g(x) * u
            fx = net_out[:, :x_dim]  # [1, 3]
            gx = net_out[:, x_dim:]  # [1, 3]

            func.u = func.dCLF(x0, x_test[i, :, :], u_test_i, fx, gx)
        else:
            func.u = u_test_i

        pred = odeint(func, x0, tt)
        pred_x = torch.cat(
            [pred_x, pred[-1, :, :].unsqueeze(0)], dim=0)
        x0 = pred[-1, :, :]
        
        if i == 23:
            print("Pred x:", x0)

        loss = torch.sum((x0 - x_test[i + 1]) ** 2)
        total_loss.append(loss.cpu().item())
        # Display loss
        print(f'timestep: {i:02d} | loss: {loss.item()}')


# Print total loss
print('total loss:   ', np.sum(total_loss))
print('Average loss: ', np.mean(total_loss))
