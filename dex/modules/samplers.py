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
            "./CBF/saved_model/NeedlePick-v0/0/CBF10.pth"))
        self.CBF.eval()
        
        self.CLF = CLF([3, 64, 6]).to(self.device)
        self.CLF.load_state_dict(torch.load(
            "./CLF/saved_model/NeedlePick-v0/0/CLF10.pth"))
        self.CLF.eval()

    def init(self):
        """Starts a new rollout. Render indicates whether output should contain image."""
        self._episode_reset()

    def sample_action(self, obs, is_train):
        return self._agent.get_action(obs, noise=is_train)

    def sample_episode(self, is_train, render=False, random_act=False):
        """Samples one episode from the environment."""
        self.init()
        episode, done = [], False

        # variable related to odeint in cbf and clf
        tt = torch.tensor([0., 0.1]).to(self.device)
        # detect constraint violation
        violate_constraint = False

        # NOTE: Must change while loop's condition back to run train.py normally
        # while not done and self._episode_step < self._max_episode_len:
        # Each step is 0.1 s, 100 steps is 10 s.
        while self._episode_step < self.cfg.max_episode_steps:
            action = self._env.action_space.sample(
            ) if random_act else self.sample_action(self._obs, is_train)
            if action is None:
                break
            if render:
                render_obs = self._env.render('rgb_array')
                img = Image.fromarray(render_obs)
                if not os.path.exists("saved_eval_pic"):
                    os.mkdir("saved_eval_pic")
                img.save(f'saved_eval_pic/image_{self._episode_step}.png')

            # ===================== CBF =====================
            if self.cfg.use_dcbf:
                # Display whether the tip of the psm has touch the obstacle or not
                # True : Collide
                # False: Safe

                # load the constraint center from the env
                constraint_center, _ = get_link_pose(self._env.obj_ids['obstacle'][0], -1)
                constraint = np.sum((self._obs['observation'][0:3] -
                                     np.array(constraint_center)) ** 2) < 0.05 ** 2
                violate_constraint = violate_constraint or constraint
                if violate_constraint:
                    print(f'warning: violate the constraint at episode step {self._episode_step}')

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
                    modified_action = self.CBF.dCBF(x0, u0, fx, g1, g2, g3, constraint_center)

                    # Remember to scale back the action before input into gym environment
                    action[0:3] = modified_action.cpu().numpy() / 0.05

            # ===================== CLF =====================
            if self.cfg.use_dclf:
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

        # make sure episode is marked as done at final time step
        episode[-1].done = True
        rollouts = self._episode_cache.pop()
        assert self._episode_step == self._max_episode_len
        return listdict2dictlist(episode), rollouts, self._episode_step

    def _episode_reset(self, global_step=None):
        """Resets sampler at the end of an episode."""
        self._episode_step, self._episode_reward = 0, 0.
        self._obs = self._reset_env()
        self._episode_cache.store_obs(self._obs)

    def _reset_env(self):
        return self._env.reset()
