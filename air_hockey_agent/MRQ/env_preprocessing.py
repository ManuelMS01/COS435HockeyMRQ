# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from collections import deque
import dataclasses
from functools import partial
from typing import Dict, Union, Tuple

import numpy as np

import utils


class ActionSpace:
    def __init__(self, shape, high, low):
        self.shape = shape
        self.high = high
        self.low = low
        self._np_random = np.random.RandomState()

    def seed(self, seed=None):
        self._np_random.seed(seed)
        return [seed]

    def sample(self):
        return self._np_random.uniform(low=self.low, high=self.high, size=self.shape)


# 1. Makes environment, sets seeds, applies wrappers.
# 2. Unifies some basic attributes like action_dim, obs_shape.
# 3. Tracks some basic information like episode timesteps and reward.
class Env:
    def __init__(self, env_name: str, seed: int=0, eval_env: bool=False, remove_info: bool=True):
        self.env = AirHockeyPreprocessing(env_name, seed, eval_env)

        # Copy instance variables
        for k in ['offline', 'pixel_obs', 'obs_shape', 'history', 'max_ep_timesteps', 'action_space']:
            self.__dict__[k] = self.env.__dict__[k]

        # Only used for printing
        self.env_name = env_name
        self.seed = seed

        self.action_space.seed(seed)
        self.discrete = self.action_space.__class__.__name__ == 'Discrete'
        self.action_dim = self.action_space.n if self.discrete else self.action_space.shape[0]
        self.max_action = 1 if self.discrete else float(self.action_space.high[0])

        self.remove_info = remove_info
        self.ep_total_reward = 0
        self.ep_timesteps = 0
        self.ep_num = 1


    def reset(self):
        self.ep_total_reward = 0
        self.ep_timesteps = 0
        self.ep_num += 1

        state, info = self.env.reset()
        return state if self.remove_info else (state, info)


    def step(self, action: Union[int, float]):
        next_state, reward, terminated, truncated, info = self.env.step(action)

        self.ep_total_reward += reward
        self.ep_timesteps += 1

        return (next_state, reward, terminated, truncated) if self.remove_info else (next_state, reward, terminated, truncated, info)


class AirHockeyPreprocessing:
    def __init__(self, env_name: str, seed: int=0, eval_env: bool=False):
        # Initialize environment parameters
        self.offline = False
        self.pixel_obs = False
        
        # Define observation and action spaces
        # These should match your air hockey environment's specifications
        self.obs_shape = (8,)  # Example: 8-dimensional state space
        self.history = 1
        self.max_ep_timesteps = 1000  # Maximum episode length
        
        # Define action space (continuous in this case)
        self.action_space = ActionSpace(
            shape=(2,),  # Example: 2-dimensional continuous action space
            high=np.array([1.0, 1.0]),  # Maximum action values
            low=np.array([-1.0, -1.0])   # Minimum action values
        )

        # Initialize history queue for state tracking
        self.history_queue = deque(maxlen=self.history)
        
        # Set random seed
        np.random.seed(seed)


    def step(self, action: Union[int, float]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple containing:
            - next_state: Next state observation
            - reward: Reward received
            - terminated: Whether the episode is terminated
            - truncated: Whether the episode is truncated
            - info: Additional information
        """
        # TODO: Implement your air hockey environment step logic here
        next_state = np.random.randn(*self.obs_shape)  # Placeholder
        reward = 0.0  # Placeholder
        terminated = False  # Placeholder
        truncated = False  # Placeholder
        info = {}  # Placeholder
        
        self.history_queue.append(next_state)
        return np.concatenate(self.history_queue), reward, terminated, truncated, info


    def reset(self) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to its initial state.
        
        Returns:
            Tuple containing:
            - state: Initial state observation
            - info: Additional information
        """
        # TODO: Implement your air hockey environment reset logic here
        state = np.random.randn(*self.obs_shape)  # Placeholder
        info = {}  # Placeholder
        
        self.history_queue.clear()
        for _ in range(self.history):
            self.history_queue.append(state)
            
        return np.concatenate(self.history_queue), info


