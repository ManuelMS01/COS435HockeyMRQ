import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
from datetime import datetime
import glob
import json

def load_config(config_path='agent_config.yml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_latest_training_dir(log_dir):
    """Get the most recent training directory"""
    training_dirs = glob.glob(os.path.join(log_dir, "training_*"))
    if not training_dirs:
        raise ValueError("No training directories found")
    return max(training_dirs, key=os.path.getctime)

def load_training_stats(training_dir):
    """Load training statistics from JSON file"""
    stats_file = os.path.join(training_dir, 'training_stats.json')
    if not os.path.exists(stats_file):
        raise ValueError(f"No training stats found in {training_dir}")
    
    with open(stats_file, 'r') as f:
        return json.load(f)

def plot_rewards(log_dir, save_plot=True):
    """
    Plot the rewards over time from the training logs.
    
    Args:
        log_dir (str): Directory containing the training logs
        save_plot (bool): Whether to save the plot to a file
    """
    # Get the latest training directory
    training_dir = get_latest_training_dir(log_dir)
    
    # Load training statistics
    stats = load_training_stats(training_dir)
    episodes = stats['episodes']
    rewards = stats['rewards']
    
    # Calculate moving average (window size = 10)
    window_size = 10
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot raw rewards
    plt.plot(episodes, rewards, 'b-', alpha=0.3, label='Raw Reward')
    
    # Plot moving average
    plt.plot(episodes[window_size-1:], moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window_size})')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.grid(True)
    plt.legend()
    
    if save_plot:
        plot_path = os.path.join(training_dir, 'training_progress.png')
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
    
    plt.show()

def main():
    # Load configuration
    config = load_config()
    
    # Plot rewards
    plot_rewards(config['log_dir'])

if __name__ == "__main__":
    main() 