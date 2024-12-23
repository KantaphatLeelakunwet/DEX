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
        # while not done and self._episode_step < self._max_episode_len:
        tt = torch.tensor([0., 0.1]).to(self.device)
        # Don't how to directly set the max episode len for gym environment
        # Leave it as 100 for now.
        # Each step is 0.1 s, 100 steps is 10 s.
        while self._episode_step < 100:
            action = self._env.action_space.sample(
            ) if random_act else self.sample_action(self._obs, is_train)
            if action is None:
                break
            if render:
                render_obs = self._env.render('rgb_array')
                img = Image.fromarray(render_obs)
                img.save(f'pic/image_{self._episode_step}.png')

            with torch.no_grad():
                if self.cfg.use_dcbf:
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
                    modified_action = self.CBF.dCBF(x0, u0, fx, g1, g2, g3)

                    # Remember to scale back the action before input into gym environment
                    action[0:3] = modified_action.cpu().numpy() / 0.05
                
                if self.cfg.use_dclf:
                    assert self.cfg.use_dcbf
                    # RL Policy
                    self.CBF.u = modified_action
                    pred_next_obs = odeint(self.CBF, x0, tt)[1, :, :]
                    temp_obs = self._obs
                    temp_obs['observation'][0:3] = pred_next_obs.cpu().numpy()
                    pred_action = self._env.action_space.sample(
                        ) if random_act else self.sample_action(temp_obs, is_train)
                    
                    
                    # ===================== CLF =====================
                    
                    # needle_rel_pos = self._obs['observation'][10:13]
                    
                    # desired_orn = [0.0, 0.0, 1.0] # vector_to_euler(needle_rel_pos)
                    # desired_orn = torch.tensor(desired_orn).unsqueeze(0).to(self.device).float()
                    
                    orn_x0 = torch.tensor(
                        self._obs['observation'][3:6]).unsqueeze(0).to(self.device).float()
                    
                    # Get desired next orientation
                    self.CLF.u = torch.tensor(pred_action[3].reshape(1, 1)).to(self.device).float()
                    desired_orn = odeint(self.CLF, orn_x0, tt)[1, :, :]
                    
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
                

            # Display whether the tip of the psm has touch the obstacle or not
            # True : Collide
            # False: Safe
            print(np.sum((self._obs['observation'][0:3] -
                  np.array([2.66255212, -0.00543937, 3.49126458])) ** 2) < 0.025 ** 2)

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
