"""
This file implements a data processing before input into ODE model.

Key functionalities:
- Load and preprocess the dataset from specified task.
- Save x, y, z position in Cartesian            ->   'obs_pos.npy'
- Save orientation in Euler and jaw angle       ->   'obs_orn.npy'
- Save position movement in Cartesian space     ->   'acs_pos.npy'
- Save orientation movement (d_yaw / d_pitch)   ->   'acs_orn.npy'
  and open status of jaw

Usage:
python data_preprocessing.py --task=NeedlePick-v0

NOTE: This code only supports tasks that only use a single PSM.
"""

import numpy as np
import argparse
import os

# # Check current working directory
# print(os.getcwd())

parser = argparse.ArgumentParser('Data Processing')
parser.add_argument(
    '--task',
    type=str,
    choices=['NeedlePick-v0', 'NeedleRegrasp-v0'],
    default='NeedlePick-v0'
)
args = parser.parse_args()

# Create a directory to store data
if not os.path.exists(args.task):
    os.makedirs(args.task)

# Load raw export demonstration data
data = np.load(
    f'../../SurRoL/surrol/data/demo/data_{args.task}_random_100.npz', allow_pickle=True)

'''
>>> data.files
['acs', 'obs', 'info']
>>> data['obs'].shape
(100, 51)
>>> data['obs'][0][0]
{'observation': array([ 2.50000024e+00,  2.50000298e-01,  3.60099983e+00, -1.57077880e+00,
       -6.55750109e-06,  1.57090934e+00, -4.43718019e-09,  2.70893839e+00,
        1.00540941e-01,  3.41057810e+00,  2.08938156e-01, -1.49459357e-01,
       -1.90421737e-01,  2.69789934e+00,  1.35331601e-01,  3.41057682e+00,
       -1.67208651e-06,  3.32018364e-06, -1.26354618e+00]), 
 'achieved_goal': array([2.69789934, 0.1353316 , 3.41057682]), 
 'desired_goal': array([2.73656776, 0.03006118, 3.576     ])}
'''

# Initialize arrays
obs_pos = np.zeros((100, 51, 3))
obs_orn = np.zeros((100, 51, 4))  # jaw angle included
acs_pos = data['acs'][:, :, 0:3]
acs_orn = data['acs'][:, :, 3]  # only d_pitch / d_yaw

# Get observation data
for i in range(100):
    for j in range(51):
        obs_pos[i, j, :] = data['obs'][i][j]['observation'][0:3]
        obs_orn[i, j, :] = data['obs'][i][j]['observation'][3:7]

# Save to npy files
np.save(f'{args.task}/obs_pos.npy', obs_pos)
np.save(f'{args.task}/obs_orn.npy', obs_orn)
np.save(f'{args.task}/acs_pos.npy', acs_pos)
np.save(f'{args.task}/acs_orn.npy', acs_orn)
