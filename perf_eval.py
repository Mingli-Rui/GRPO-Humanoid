import argparse
import datetime
import os
import time
import random

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from lib.GRPOAgent import GRPOAgent
from lib.GRPOBuffer2 import GRPOBuffer2


def get_perf(env, agent, device, seed):
    """
    Log a video of one episode of the agent playing in the environment.
    :param env: a test environment which supports video recording and doesn't conflict with the other environments.
    :param agent: the agent to record.
    :param device: the device to run the agent on.
    :param video_path: the path to save the video.
    :param fps: the frames per second of the video.
    """
    # try 5 times, save the longest one
    traj_count = 49
    traj_rewards = []
    traj_lens = []
    with torch.no_grad():
        for i in range(traj_count):
            total_reward = 0.0
            traj_len = 0
            obs, _ = env.reset(seed=seed)
            done = False
            while not done:
                # Compute an action
                action, _, _, _ = agent.get_action_and_value(
                        torch.tensor(np.array([obs], dtype=np.float32), device=device))
                # Step the environment
                obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
                total_reward += reward
                traj_len += 1
                done = terminated or truncated
            traj_rewards.append(total_reward)
            traj_lens.append(traj_len)

    # print("rewards stats:", np.mean(traj_rewards), np.median(traj_rewards), np.std(traj_rewards))
    print(f"count: {len(traj_lens)}, length status", np.mean(traj_lens), np.median(traj_lens), np.std(traj_lens))
    return np.mean(traj_rewards), np.median(traj_rewards), np.std(traj_rewards)


def make_env(env_id, n_steps, reward_scaling=1.0, render=False, fps=30):
    """
    Make an environment with the given id.
    :param env_id: the id of the environment.
    :param reward_scaling: the scaling factor for the rewards.
    :param render: whether to render the environment.
    :param fps: the frames per second if rendering.
    :return: the environment.
    """
    if render:
        env = gym.make(env_id, render_mode='rgb_array', max_episode_steps=n_steps)
        env.metadata['render_fps'] = fps
        env = gym.wrappers.TransformReward(env, lambda r: r * reward_scaling)
    else:
        env = gym.make(env_id, max_episode_steps=n_steps)
        env = gym.wrappers.TransformReward(env, lambda r: r * reward_scaling)
    return env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True, help="Name of the run")
    parser.add_argument("--env", default="Humanoid-v4", help="Environment to use")
    parser.add_argument("--n-steps", type=int, default=1024, help="max number of steps per trajectory")
    parser.add_argument("--reward-scale", type=float, default=0.005, help="Reward scaling")
    # parser.add_argument("--model", required=True, help="path of the model")
    parser.add_argument("--number", type=int, default=5, help="Number of video")
    parser.add_argument("--seed", type=int, default=1, help="seed")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Support Mac mps
    # device_name = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    # TODO use cpu for simplicity.
    # if device_name != "cuda" and torch.backends.mps.is_available():
    #     device_name = 'mps'
    # print(f'train on device: {device_name}, beta: {args.beta}, seed={args.seed}')
    device = torch.device('cpu')

    # Create the folders for logging
    current_dir = os.path.dirname(__file__)
    folder_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.run_name}"
    videos_dir = os.path.join(current_dir, "videos", folder_name)
    os.makedirs(videos_dir, exist_ok=True)

    # hardcode for model file
    first_baseline = [
        ("checkpoints/2025-03-14_13-48-20_run_sim_1.0_average/model.dat", 7048.19 / 60, "sigma=1.0,average"),
        ("checkpoints/2025-03-14_15-47-18_run_sim_1.0_max_return/model.dat", 6625.48 / 60, "sigma=1.0,max_return"),
        ("checkpoints/2025-03-14_11-51-46_run_sim_1.0_median/model.dat", 6877.07 / 60, "sigma=1.0,median"),
        ("checkpoints/2025-03-14_17-40-18_run_sim_2.0_median/model.dat", 6877.07 / 60, "sigma=2.0,median"),
        ("checkpoints/2025-03-14_19-36-48_run_sim_0.5_median/model.dat", 6877.07 / 60, "sigma=0.5,median"),
    ]

    sigma_baseline = [
        ("checkpoints/2025-03-15_12-57-37_run_sim_0.5_max_return/model.dat", 6625.48 / 60, "sigma=0.5,max_return"),
        ("checkpoints/2025-03-15_14-45-37_run_sim_1.0_max_return/model.dat", 6877.07 / 60, "sigma=1.0,max_return"),
        ("checkpoints/2025-03-15_16-37-44_run_sim_1.5_max_return/model.dat", 6877.07 / 60, "sigma=1.5,max_return"),
        ("checkpoints/2025-03-15_18-29-09_run_sim_2.0_max_return/model.dat", 6877.07 / 60, "sigma=2.0,max_return"),
        ("checkpoints/2025-03-15_20-19-41_run_sim_2.5_max_return/model.dat", 6877.07 / 60, "sigma=2.5,max_return"),
    ]

    model_paths = sigma_baseline

    results = []  # average, median, std, wall-time
    for model_path, wall_time, name in model_paths:
        # Create the environments
        test_env = make_env(args.env, args.n_steps, reward_scaling=args.reward_scale, render=False)
        obs_dim = test_env.observation_space.shape
        act_dim = test_env.action_space.shape

        # Create the agent and optimizer
        agent = GRPOAgent(obs_dim[0], act_dim[0]).to(device)
        agent.load_state_dict(torch.load(model_path))
        agent.eval()

        print(agent.actor_mu)
        print(agent.actor_logstd)

        # set seed except env
        random.seed(args.seed)
        np.random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        torch.manual_seed(args.seed)

        try:
            mean, median, std = get_perf(test_env, agent, device, args.seed)
            results.append((name, mean, median, std, wall_time))

        finally:
            # Close the environments and tensorboard writer
            test_env.close()

    print(results)

    # Define colors for better visualization
    colors_primary = ['#1f77b4',  # Blue
                      '#ff7f0e',  # Orange
                      '#2ca02c',  # Green
                      '#d62728',  # Red
                      '#9467bd']  # Purple
    fig, ax1 = plt.subplots(figsize=(12, 7))  # Keep a large figure for clarity

    # Number of results
    num_results = len(results)
    colors_primary = colors_primary[0:num_results]
    bar_width = 0.15  # Maintain a good bar width
    gap = 0.1  # Reduce gap between metric groups

    # Labels for metrics (excluding Wall Time)
    x_labels = ['Mean', 'Median', 'Std']
    x_positions = np.arange(len(x_labels)) * (bar_width * num_results + gap)  # Reduce space between groups

    # Extracting values (excluding Wall Time)
    names, means, medians, stds, _ = zip(*results)

    # Plot mean, median, and std on primary y-axis with distinct colors
    for i in range(num_results):
        ax1.bar(x_positions + i * bar_width, [means[i], medians[i], stds[i]],
                bar_width, label=f'{names[i]}', color=colors_primary[i])

    # Set labels and title for primary y-axis
    # ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Returns (Mean, Median, Std Dev)')
    ax1.set_title('Returns comparison for hyperparameters (500 epochs)')
    ax1.set_xticks(x_positions + (num_results / 2 - 0.5) * bar_width)
    ax1.set_xticklabels(x_labels)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust legend position to be directly above the "Std Dev" metric
    ax1.legend(loc='upper left', fontsize=14, bbox_to_anchor=(0.65, 1.02))

    # Show the plot
    plt.show()
