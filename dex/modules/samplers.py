from ..utils.general_utils import AttrDict, listdict2dictlist
from ..utils.rl_utils import ReplayCache

import sys
sys.path.append('./CBF')
from CBF.cbf import CBF
sys.path.append('./CLF')
from CLF.clf import CLF

import os
import torch
from torchdiffeq import odeint
import numpy as np
import PIL.Image as Image
from vec2orn import vector_to_euler
import copy

from surrol.utils.pybullet_utils import (
    get_link_pose,
)

import pybullet as p
from scipy.spatial.transform import Rotation


class Sampler:
    """Collects rollouts from the environment using the given agent."""

    def __init__(self, env, agent, max_episode_len, config):
        self._env = env
        self._agent = agent
        self._max_episode_len = max_episode_len
        self.cfg = config

        self._obs = None
        self._episode_step = 0
        self._episode_cache = ReplayCache(max_episode_len)

        self.device = torch.device(
            'cuda:' + str(0)
            if torch.cuda.is_available() else 'cpu'
        )

        # Initialize neuralODE for CBF (Evaluation ONLY)
        self.CBF = CBF([3, 64, 12]).to(self.device)
        self.CBF.load_state_dict(torch.load(
            f"./CBF/saved_model/{self.cfg.task[0:-1]}0/0/CBF10.pth"))
        self.CBF.eval()
        
        self.CLF = CLF([3, 64, 6]).to(self.device)
        self.CLF.load_state_dict(torch.load(
            f"./CLF/saved_model/{self.cfg.task[0:-1]}0/0/CLF10.pth"))
        self.CLF.eval()

        self.dcbf_constraint_type = int(self.cfg.task[-1])
        print(f'constraint type is {self.dcbf_constraint_type}')

    def init(self):
        """Starts a new rollout. Render indicates whether output should contain image."""
        self._episode_reset()

    def sample_action(self, obs, is_train):
        return self._agent.get_action(obs, noise=is_train)

    def sample_episode(self, is_train, ep, render=False, random_act=False, render_three_views=False):
        """Samples one episode from the environment."""
        self.init()
        episode, done = [], False

        # variable related to odeint in cbf and clf
        tt = torch.tensor([0., 0.1]).to(self.device)
        
        # Store number of violations
        num_violations = 0

        # NOTE: Must change while loop's condition back to run train.py normally
        # while not done and self._episode_step < self._max_episode_len:
        # Each step is 0.1 s, 100 steps is 10 s.
        while self._episode_step < self.cfg.max_episode_steps:
            action = self._env.action_space.sample(
            ) if random_act else self.sample_action(self._obs, is_train)
            if action is None:
                break
            if render:
                if render_three_views:
                    front_rgb_array, right_rgb_array, top_rgb_array = self._env.render_three_views('rgb_array')
                    render_obs = np.concatenate([front_rgb_array, right_rgb_array, top_rgb_array], axis=1)
                else:
                    render_obs = self._env.render('rgb_array')

                img = Image.fromarray(render_obs)
                if self.cfg.use_dclf:
                    if not os.path.exists(f"saved_eval_pic/CLF/{self.cfg.task}/{ep:02}"):
                        os.mkdir(f"saved_eval_pic/CLF/{self.cfg.task}/{ep:02}")
                    img.save(f'saved_eval_pic/CLF/{self.cfg.task}/{ep:02}/image_{self._episode_step}.png')
                    # print("Saved at", f'saved_eval_pic/CLF/{self.cfg.task}/{ep:02}/image_{self._episode_step}.png')
                elif self.cfg.use_dcbf:
                    if not os.path.exists(f"saved_eval_pic/CBF/{self.cfg.task}/{ep:02}"):
                        os.mkdir(f"saved_eval_pic/CBF/{self.cfg.task}/{ep:02}")
                    img.save(f'saved_eval_pic/CBF/{self.cfg.task}/{ep:02}/image_{self._episode_step}.png')
                    # print("Saved at", f'saved_eval_pic/CBF/{self.cfg.task}/{ep:02}/image_{self._episode_step}.png')
                else:
                    if not os.path.exists(f"saved_eval_pic/NONE/{self.cfg.task}/{ep:02}"):
                        os.mkdir(f"saved_eval_pic/NONE/{self.cfg.task}/{ep:02}")
                    img.save(f'saved_eval_pic/NONE/{self.cfg.task}/{ep:02}/image_{self._episode_step}.png')
                    # print("Saved at", f'saved_eval_pic/NONE/{self.cfg.task}/{ep:02}/image_{self._episode_step}.png')
                # if not os.path.exists("saved_eval_pic"):
                #     os.mkdir("saved_eval_pic")
                # img.save(f'saved_eval_pic/image_{self._episode_step}.png')

            # ===================== constraint test =====================
            # Display whether the tip of the psm has touch the obstacle or not
            # True : Collide
            # False: Safe
            if self.dcbf_constraint_type == 1:
                # sphere constraint
                # load the constraint center from the env
                constraint_center, _ = get_link_pose(self._env.obj_ids['obstacle'][0], -1)
                violate_constraint = self.CBF.constraint_valid(constraint_type=self.dcbf_constraint_type,
                                                               robot=self._obs['observation'][0:3],
                                                               constraint_center=constraint_center)
            elif self.dcbf_constraint_type == 2:
                # surface constraint
                point, surface_ori = get_link_pose(self._env.obj_ids['obstacle'][0], -1)
                rot_matrix = Rotation.from_quat(np.array(surface_ori)).as_matrix()
                # print(rot.as_euler('xyz'))
                original_normal_vector = np.array([0, 1, 0]).reshape([3, 1])
                normal_vector = (rot_matrix @ original_normal_vector).reshape(-1).tolist()
                # print(normal_vector)
                violate_constraint = self.CBF.constraint_valid(constraint_type=self.dcbf_constraint_type,
                                                               robot=self._obs['observation'][0:3], point=point,
                                                               normal_vector=normal_vector)
            elif self.dcbf_constraint_type == 3:
                # box constraint
                point, box_ori = get_link_pose(self._env.obj_ids['obstacle'][0], -1)
                rot = Rotation.from_quat(np.array(box_ori))
                inv_matrix = rot.inv().as_matrix()
                original_robot_pos = np.array(self._obs['observation'][0:3]).reshape(3, 1)
                inv_relative_robot_pos = inv_matrix @ (original_robot_pos-np.array(point).reshape(3, 1))
                box_size = np.array([0.25, 0.25, 0.25])

                box_min = -box_size/2
                box_max = box_size/2
                # print(box_max)
                # print(box_min)
                # print(self._obs['observation'][0:3])
                violate_constraint = self.CBF.constraint_valid(constraint_type=self.dcbf_constraint_type,
                                                               robot=inv_relative_robot_pos.reshape(-1).tolist(),
                                                               box_min=box_min.tolist(),
                                                               box_max=box_max.tolist())
            elif self.dcbf_constraint_type == 4:
                # half-sphere constraint
                center, sphere_ori = get_link_pose(self._env.obj_ids['obstacle'][0], -1)
                if self.cfg.task == 'PegTransfer-v4':
                    radius = 0.1
                else:
                    radius = 0.05
                rot_matrix = Rotation.from_quat(np.array(sphere_ori)).as_matrix()
                original_normal_vector = np.array([0, 1, 0]).reshape([3, 1])
                normal_vector = (rot_matrix @ original_normal_vector).reshape(-1).tolist()
                # for area 0, the agent can stay in area 0 or go to area 1 or area 2
                # for area 1, the agent can only stay in area 1 or go to area 0
                # for area 2, the agent can only stay in area 2 or go to area 0
                if np.dot(normal_vector, np.array(self._obs['observation'][0:3])-np.array(center)) < 0:
                    out = self.CBF.constraint_valid(constraint_type=self.dcbf_constraint_type,
                                                        robot=self._obs['observation'][0:3],
                                                        constraint_center=center,
                                                        radius=radius)
                    if out:
                        current_area = 1
                    else:
                        current_area = 2
                else:
                    current_area = 0
                
                # print(current_area)
                
                if self._episode_step == 0:
                    violate_constraint = False
                else:
                    if last_area + current_area == 3:
                        # last and current areas are (1, 2) or (2, 1)
                        violate_constraint = True
                    else:
                        violate_constraint = False
                last_area = current_area
            elif self.dcbf_constraint_type == 5:
                # cylinder constraint
                center, cylinder_ori = get_link_pose(self._env.obj_ids['obstacle'][0], -1)
                cylinder_length = 0.35
                radius = 0.06
                rot_matrix = Rotation.from_quat(np.array(cylinder_ori)).as_matrix()
                original_ori_vector = np.array([0, 0, 1]).reshape([3, 1])
                current_ori_vector = (rot_matrix @ original_ori_vector).reshape(-1).tolist()
                proj_vec = np.dot(current_ori_vector, np.array(self._obs['observation'][0:3])-np.array(center))
                # for area 0, the agent can stay in area 0 or go to area 1 or area 2
                # for area 1, the agent can only stay in area 1 or go to area 0
                # for area 2, the agent can only stay in area 2 or go to area 0
                if proj_vec**2 < (cylinder_length/2)**2:
                    out = self.CBF.constraint_valid(constraint_type=self.dcbf_constraint_type,
                                                        robot=self._obs['observation'][0:3],
                                                        constraint_center=center, radius=radius,
                                                        ori_vector=current_ori_vector)
                    if out:
                        current_area = 1
                    else:
                        current_area = 2
                else:
                    current_area = 0
                if self._episode_step == 0:
                    violate_constraint = False
                else:
                    if last_area + current_area == 3:
                        # last and current areas are (1, 2) or (2, 1)
                        violate_constraint = True
                    else:
                        violate_constraint = False
                last_area = current_area
            elif self.dcbf_constraint_type == 6:
                # complex cylinder constraint
                center, cylinder_ori = get_link_pose(self._env.obj_ids['obstacle'][0], -1)
                cylinder_length = 0.1
                radius = 0.05
                total_length = cylinder_length*2+radius*2
                # discretize the center line of the cylinder and store those discretized points
                self.CBF.build_discretized_center_line(cylinder_length, radius, center, cylinder_ori)
                rot_matrix = Rotation.from_quat(np.array(cylinder_ori)).as_matrix()
                original_ori_vector = np.array([0, 0, 1]).reshape([3, 1])
                current_ori_vector = (rot_matrix @ original_ori_vector).reshape(-1).tolist()
                proj_vec = np.dot(current_ori_vector, np.array(self._obs['observation'][0:3])-np.array(center))
                # for area 0, the agent can stay in area 0 or go to area 1 or area 2
                # for area 1, the agent can only stay in area 1 or go to area 0
                # for area 2, the agent can only stay in area 2 or go to area 0
                if proj_vec**2 < (total_length/2)**2:
                    out = self.CBF.constraint_valid(constraint_type=self.dcbf_constraint_type,
                                                        robot=self._obs['observation'][0:3], radius=radius)
                    if out:
                        current_area = 1
                    else:
                        current_area = 2
                else:
                    current_area = 0
                if self._episode_step == 0:
                    violate_constraint = False
                else:
                    if last_area + current_area == 3:
                        # last and current areas are (1, 2) or (2, 1)
                        violate_constraint = True
                    else:
                        violate_constraint = False
                last_area = current_area
            else:
                violate_constraint = False
                
            if violate_constraint:
                num_violations += 1
                print(f'Episode {ep:02}: warning: violate the constraint at episode step {self._episode_step}')

            # ===================== CBF =====================
            isModified = False
            if self.cfg.use_dcbf and self.dcbf_constraint_type != 0:
                with torch.no_grad():
                    x0 = torch.tensor(
                        self._obs['observation'][0:3]).unsqueeze(0).to(self.device).float()

                    # 0.05 is scaling for needlepick only
                    u0 = 0.05 * \
                        torch.tensor(action[0:3]).unsqueeze(0).to(self.device).float()

                    cbf_out = self.CBF.net(x0)  # [1, 12]

                    # \dot{x} = f(x) + g(x) * u
                    fx = cbf_out[:, :3]  # [1, 3]
                    gx = cbf_out[:, 3:]  # [1, 9]

                    g1, g2, g3 = torch.chunk(gx, 3, dim=-1)  # [1, 3]
                    if self.dcbf_constraint_type == 1:
                        modified_action = self.CBF.dCBF_sphere(x0, u0, fx, g1, g2, g3, constraint_center)
                    elif self.dcbf_constraint_type == 2:
                        modified_action = self.CBF.dCBF_surface(x0, u0, fx, g1, g2, g3, point, normal_vector)
                    elif self.dcbf_constraint_type == 3:
                        modified_action = self.CBF.dCBF_box(x0, u0, fx, g1, g2, g3, box_size, inv_matrix, np.array(point))
                    elif self.dcbf_constraint_type == 4:
                        if current_area > 0:
                            modified_action = self.CBF.dCBF_half_sphere(x0, u0, fx, g1, g2, g3,
                                                                        center, radius, current_area)
                        else:
                            modified_action = torch.tensor(action[0:3]).to(self.device)*0.05
                    elif self.dcbf_constraint_type == 5:
                        if current_area > 0:
                            modified_action = self.CBF.dCBF_cylinder(x0, u0, fx, g1, g2, g3,
                                                                     current_ori_vector, center, radius, current_area)
                        else:
                            modified_action = torch.tensor(action[0:3]).to(self.device)*0.05
                    elif self.dcbf_constraint_type == 6:
                        print(current_area)
                        if current_area > 0:
                            modified_action = self.CBF.dCBF_complex_cylinder(x0, u0, fx, g1, g2, g3,
                                                                             radius, current_area)
                        else:
                            modified_action = torch.tensor(action[0:3]).to(self.device)*0.05

                    isModified = True
                    # Check if action is modified by CBF
                    if (modified_action.cpu().numpy() == 0.05 * action[0:3]).all():
                        # print("ACTION IS NOT MODIFIED!!!")
                        isModified = False
                    
                    # Remember to scale back the action before input into gym environment
                    action[0:3] = modified_action.cpu().numpy() / 0.05

            # ===================== CLF =====================
            if isModified and self.cfg.use_dclf and self.dcbf_constraint_type != 0:
                assert self.cfg.use_dcbf
                with torch.no_grad():
                    # predicted next position given the modified action
                    self.CBF.u = modified_action
                    pred_next_position = odeint(self.CBF, x0, tt)[1, :, :]

                # ------------get desired orientation------------
                # use predicted next position and the critic to get the desired orientation

                # Get initial guess for orientation
                with torch.no_grad():
                    orn_x0 = torch.tensor(
                        self._obs['observation'][3:6]).unsqueeze(0).to(self.device).float()
                    self.CLF.u = torch.tensor(action[3].reshape(1, 1)).to(self.device).float()
                    update_orn = odeint(self.CLF, orn_x0, tt)[1, 0, :]
                # update_orn = torch.tensor(self._obs['observation'][3:6]).cuda().float()
                for _ in range(10):
                    o = torch.tensor(self._obs['observation']).reshape(1, -1).cuda().float()
                    g = torch.tensor(self._obs['desired_goal']).reshape(1, -1).cuda().float()
                    o[:, 0:3] = pred_next_position
                    update_orn.requires_grad = True
                    o[:, 3:6] = update_orn

                    # calculate gradient of the critic with respect to the orientation
                    input_tensor = self._agent._preproc_inputs(o, g, device='cuda')
                    predicted_next_action = self._agent.actor(input_tensor)
                    value = self._agent.critic_target(input_tensor, predicted_next_action)
                    value.backward()

                    update_grad = update_orn.grad.clone().detach()

                    # update the orientation with the gradient
                    step_size = 0.001
                    with torch.no_grad():
                        updated_orn = update_orn+update_grad*step_size

                    # test the updated orn
                    with torch.no_grad():
                        o[:, 3:6] = updated_orn
                        input_tensor = self._agent._preproc_inputs(o, g, device='cuda')
                        predicted_next_action = self._agent.actor(input_tensor)
                        value_new = self._agent.critic_target(input_tensor, predicted_next_action)
                        if value_new.item() > value.item():
                            # print(f'before update: {value.item()}')
                            # print(f'before update: {update_orn}')
                            # print(f'after update: {value_new.item()}')
                            # print(f'after update: {updated_orn}')
                            update_orn = updated_orn.clone().detach()
                        else:
                            break
                desired_orn = update_orn.clone().detach().unsqueeze(0)

                # use fixed desired orientation
                # desired_orn = [0.0, 0.0, 1.0]  # vector_to_euler(needle_rel_pos)
                # desired_orn = torch.tensor(desired_orn).unsqueeze(0).to(self.device).float()

                # use waypoint orientation
                # desired_orn = torch.tensor(self._obs['observation'][-3:]).unsqueeze(0).to(self.device).float()

                # use rl policy to predict the desired orientation
                # temp_obs = copy.deepcopy(self._obs)
                # temp_obs['observation'][0:3] = pred_next_position.cpu().numpy()
                # pred_action = self._env.action_space.sample(
                #     ) if random_act else self.sample_action(temp_obs, is_train)
                #
                # # Get desired next orientation
                # self.CLF.u = torch.tensor(pred_action[3].reshape(1, 1)).to(self.device).float()
                # desired_orn = odeint(self.CLF, orn_x0, tt)[1, :, :]

                # ------------use desired orientation------------
                with torch.no_grad():
                    orn_x0 = torch.tensor(
                        self._obs['observation'][3:6]).unsqueeze(0).to(self.device).float()

                    # 0.05 is scaling for needlepick only
                    orn_u0 = np.deg2rad(30) * \
                        torch.tensor(action[3]).unsqueeze(0).to(self.device).float()

                    clf_out = self.CLF.net(orn_x0)  # [1, 6]

                    # \dot{x} = f(x) + g(x) * u
                    fx = clf_out[:, :3]  # [1, 3]
                    gx = clf_out[:, 3:]  # [1, 3]

                    modified_orn = self.CLF.dCLF(orn_x0, desired_orn, orn_u0, fx, gx)

                    # Remember to scale back the action before input into gym environment
                    action[3] = modified_orn.cpu().numpy() / np.deg2rad(30)


            obs, reward, done, info = self._env.step(action)
            # print(info)
            episode.append(AttrDict(
                reward=reward,
                success=info['is_success'],
                info=info
            ))
            self._episode_cache.store_transition(obs, action, done)
            if render:
                episode[-1].update(AttrDict(image=render_obs))

            # update stored observation
            self._obs = obs
            self._episode_step += 1

        if episode[-1]['success'] == 1.0:
            if self.cfg.use_dclf:
                success_filename = f"saved_eval_pic/CLF/{self.cfg.task}/{ep:02}/success.txt"
            elif self.cfg.use_dcbf:
                success_filename = f"saved_eval_pic/CBF/{self.cfg.task}/{ep:02}/success.txt"
            else:
                success_filename = f"saved_eval_pic/NONE/{self.cfg.task}/{ep:02}/success.txt"
            with open(success_filename, 'w') as file:
                file.write("Hello, World!\n")
            file.close()
        # make sure episode is marked as done at final time step
        episode[-1].done = True
        rollouts = self._episode_cache.pop()
        assert self._episode_step == self._max_episode_len
        return listdict2dictlist(episode), rollouts, self._episode_step, num_violations

    def _episode_reset(self, global_step=None):
        """Resets sampler at the end of an episode."""
        self._episode_step, self._episode_reward = 0, 0.
        self._obs = self._reset_env()
        self._episode_cache.store_obs(self._obs)

    def _reset_env(self):
        return self._env.reset()
