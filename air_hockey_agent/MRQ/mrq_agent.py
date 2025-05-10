import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mushroom_rl.core import Agent
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.algorithms.value import DQN
from mushroom_rl.utils.replay_memory import ReplayMemory

class MRQNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()
        
        # Input shape should be (n_states,)
        # The environment actually provides 46 dimensions, but reports 23
        n_input = 46  # Fixed input size for the air hockey environment
        
        # Network architecture for value function
        self.h1 = nn.Linear(n_input, 256)
        self.h2 = nn.Linear(256, 256)
        # Output a single value estimate
        self.h3 = nn.Linear(256, 1)
        
    def forward(self, state, action=None):
        # Ensure state is a tensor and convert to float32
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state)
        state = state.float()  # Convert to float32
        
        # Process state through network
        x = F.relu(self.h1(state))
        x = F.relu(self.h2(x))
        q = self.h3(x)  # Output value estimate
        
        return q

class MRQAgent(Agent):
    def __init__(self, mdp_info, **kwargs):
        # Get parameters from kwargs or use defaults
        self._lr = kwargs.get('lr', 0.001)
        self._batch_size = kwargs.get('batch_size', 32)
        self._replay_memory_size = kwargs.get('replay_memory_size', 10000)
        self._target_update_frequency = kwargs.get('target_update_frequency', 1000)
        self._initial_replay_size = kwargs.get('initial_replay_size', 500)
        self._max_replay_size = kwargs.get('max_replay_size', 50000)
        self._epsilon = kwargs.get('epsilon', 0.1)
        
        # Create the approximator with fixed input size
        approximator_params = dict(
            network=MRQNetwork,
            optimizer={'class': torch.optim.Adam,
                      'params': {'lr': self._lr}},
            loss=F.mse_loss,
            input_shape=(46,),  # Fixed input size for the air hockey environment
            output_shape=(1,),  # Single value estimate
            n_actions=1,
            **kwargs
        )
        
        # Create the policy
        policy = EpsGreedy(epsilon=self._epsilon)
        
        # Create the replay memory with fixed input size
        self._replay_memory = ReplayMemory(
            self._replay_memory_size,
            46  # Fixed input size for the air hockey environment
        )
        
        # Create the DQN algorithm
        self._dqn = DQN(
            mdp_info,
            policy,
            TorchApproximator,
            approximator_params=approximator_params,
            batch_size=self._batch_size,
            initial_replay_size=self._initial_replay_size,
            max_replay_size=self._max_replay_size,
            target_update_frequency=self._target_update_frequency
        )
        
        super().__init__(mdp_info, policy, self._dqn.approximator)
        
    def fit(self, dataset):
        # Process the dataset for value function learning
        value_dataset = []
        
        for state, action, reward, next_state, absorbing, last in dataset:
            # Ensure state and next_state are numpy arrays
            state_np = np.array(state)
            next_state_np = np.array(next_state)
            
            # For value function learning, we just need state, reward, next_state
            value_dataset.append((state_np, None, reward, next_state_np, absorbing, last))
        
        # Fit the DQN with the value dataset
        self._dqn.fit(value_dataset)
        
    def draw_action(self, state):
        # Ensure state has the correct shape
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        # Get value estimate from the network
        value = self._dqn.approximator.predict(state)
        
        # For now, return random actions within bounds
        # TODO: Implement proper action selection based on value estimates
        action1 = np.random.uniform(-1, 1, size=(2, 7))
        action2 = np.random.uniform(-1, 1, size=(2, 7))
        
        return (action1, action2)
        
    def episode_start(self):
        self._dqn.episode_start()
        
    def stop(self):
        self._dqn.stop() 