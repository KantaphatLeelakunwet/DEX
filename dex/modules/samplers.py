from ..utils.general_utils import AttrDict, listdict2dictlist
from ..utils.rl_utils import ReplayCache
from .cbf import ODEFunc

import os
import torch
import PIL.Image as Image


class Sampler:
    """Collects rollouts from the environment using the given agent."""

    def __init__(self, env, agent, max_episode_len):
        self._env = env
        self._agent = agent
        self._max_episode_len = max_episode_len

        self._obs = None
        self._episode_step = 0
        self._episode_cache = ReplayCache(max_episode_len)

        self.device = torch.device(
            'cuda:' + str(0)
            if torch.cuda.is_available() else 'cpu'
        )

        # Initialize neuralODE for CBF (Evaluation ONLY)
        self.func = ODEFunc([3, 64, 12]).to(self.device)
        self.func.load_state_dict(torch.load("./dex/modules/model_test10.pth"))
        self.func.eval()

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
