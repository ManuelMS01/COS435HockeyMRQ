from air_hockey_agent.MRQ.MRQ import Agent
from air_hockey_challenge.framework.evaluate_agent import evaluate
import torch

def build_agent(env_info, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    Args:
        env_info (dict): The environment information
        kwargs (any): Additionally setting from agent_config.yml
    Returns:
         (AgentBase) An instance of the Agent
    """
    # Extract environment information
    obs_shape = env_info['observation_space'].shape
    action_dim = env_info['action_space'].shape[0]
    max_action = float(env_info['action_space'].high[0])
    
    # Initialize the agent with continuous action space settings
    agent = Agent(
        obs_shape=obs_shape,
        action_dim=action_dim,
        max_action=max_action,
        pixel_obs=False,  # Air hockey uses state observations, not pixels
        discrete=False,   # Continuous action space
        device=torch.device('cpu'),  # Using CPU since we're on Mac
        history=1,        # Number of past states to consider
        hp={}            # Use default hyperparameters
    )
    
    return agent
