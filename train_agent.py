import os
import torch
import numpy as np
from air_hockey_agent.MRQ.mrq_agent import MRQAgent
from air_hockey_challenge.framework.evaluate_agent import evaluate
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from mushroom_rl.environments import Gym
import yaml
from datetime import datetime
import json

def load_config(config_path='/Users/manuelmartinez/air_hockey_challenge/air_hockey_challenge/air_hockey_agent/agent_config.yml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def create_log_dir(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def save_training_stats(log_dir, episode, reward):
    """Save training statistics to a JSON file"""
    stats_file = os.path.join(log_dir, 'training_stats.json')
    
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    else:
        stats = {'episodes': [], 'rewards': []}
    
    stats['episodes'].append(episode)
    stats['rewards'].append(float(reward))
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f)

def train_agent(config):
    # Create log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config['log_dir'], f"training_{timestamp}")
    create_log_dir(log_dir)
    
    # Create environment
    env = AirHockeyChallengeWrapper(
        env=config['env'][0],  
        interpolation_order=config['interpolation_order']
    )
    
    # Create agent
    agent = MRQAgent(
        mdp_info=env.info,  # Use the environment's info object instead of env_info dictionary
        lr=0.001,
        batch_size=32,
        replay_memory_size=10000,
        epsilon=0.1
    )
    
    # Training parameters
    n_episodes = config.get('n_episodes', 1000)  # Number of training episodes
    save_interval = 100  # Save model every 100 episodes
    
    # Training loop
    for episode in range(n_episodes):
        # Run a single episode
        state = env.reset()
        done = False
        episode_reward = 0
        dataset = []
        
        while not done:
            # Get action from agent
            print(f"State shape before reshaping: {state.shape}")
            state = state.reshape(1, -1)
            print(f"State shape after reshaping: {state.shape}")
            action = agent.draw_action(state)
            
            # Print action shapes for debugging
            print(f"Action shapes: {[a.shape for a in action]}")
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition in DQN format
            # (state, action, reward, next_state, absorbing, last)
            # absorbing is True if the episode is done
            # last is True if this is the last step of the episode
            dataset.append((state, action, reward, next_state, done, done))
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Update agent
            if len(dataset) >= agent._batch_size:
                agent.fit(dataset[-agent._batch_size:])
        
        # Save model and stats periodically
        if (episode + 1) % save_interval == 0:
            model_path = os.path.join(log_dir, f'model_episode_{episode+1}.pth')
            torch.save(agent._dqn.approximator.model.state_dict(), model_path)
            print(f"Model saved at episode {episode+1}")
        
        # Save training statistics
        save_training_stats(log_dir, episode + 1, episode_reward)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{n_episodes}, Reward: {episode_reward:.2f}")
    
    # Save final model
    final_model_path = os.path.join(log_dir, 'final_model.pth')
    torch.save(agent._dqn.approximator.model.state_dict(), final_model_path)
    print(f"Training completed. Final model saved to {final_model_path}")
    
    return agent

def test_agent(agent, config):
    # Create test environment
    env = AirHockeyChallengeWrapper(
        env=config['env'][0],  # Use the first environment from the config
        interpolation_order=config['interpolation_order']
    )
    
    # Run test episodes
    print("\nTesting agent...")
    test_episodes = 10
    total_reward = 0
    
    for episode in range(test_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.draw_action(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        
        total_reward += episode_reward
        print(f"Test episode {episode+1}: Reward = {episode_reward}")
    
    avg_reward = total_reward / test_episodes
    print(f"\nAverage test reward: {avg_reward}")

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Train agent
    trained_agent = train_agent(config)
    
    # Test agent
    test_agent(trained_agent, config) 