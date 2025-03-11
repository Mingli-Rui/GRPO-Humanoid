import argparse
import datetime
import os
import time
import random

import cv2
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import umap
import hdbscan
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from lib.GRPOAgent import GRPOAgent
from lib.GRPOBuffer2 import GRPOBuffer2


def make_env(env_id, reward_scaling=1.0, render=False, fps=30):
    """
    Make an environment with the given id.
    :param env_id: the id of the environment.
    :param reward_scaling: the scaling factor for the rewards.
    :param render: whether to render the environment.
    :param fps: the frames per second if rendering.
    :return: the environment.
    """
    if render:
        env = gym.make(env_id, render_mode='rgb_array')
        env.metadata['render_fps'] = fps
        env = gym.wrappers.TransformReward(env, lambda r: r * reward_scaling)
    else:
        env = gym.make(env_id)
        env = gym.wrappers.TransformReward(env, lambda r: r * reward_scaling)
    return env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True, help="Name of the run")
    parser.add_argument("--env", default="Humanoid-v4", help="Environment to use")
    parser.add_argument("--reward-scale", type=float, default=0.005, help="Reward scaling")
    parser.add_argument("--model", required=True, help="path of the model")
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
    videos_dir = os.path.join(current_dir, "photos", folder_name)
    os.makedirs(videos_dir, exist_ok=True)

    # Create the environments
    test_env = make_env(args.env, reward_scaling=args.reward_scale, render=True)
    obs_dim = test_env.observation_space.shape
    act_dim = test_env.action_space.shape

    # Create the agent and optimizer
    agent = GRPOAgent(obs_dim[0], act_dim[0]).to(device)
    agent.load_state_dict(torch.load(args.model))
    agent.eval()

    print(agent.actor_mu)
    print(agent.actor_logstd)

    # set seed except env
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)

    try:
        obs, _ = test_env.reset(seed=args.seed)
        size = args.number
        size = 48 * 1024

        # Row out observations for the specific model
        X = np.zeros((size, obs_dim[0]), dtype=float)
        for i in range(size):
            # Sample the actions
            with torch.no_grad():
                actions, logprobs, _, values = agent.get_action_and_value(torch.tensor(obs, dtype=torch.float32))
                # values = values.flatten()

            # Step the environment
            next_obs, _, next_terminateds, next_truncateds, _ = test_env.step(actions.cpu().numpy())
            X[i] = obs
            obs = next_obs
            if next_terminateds or next_truncateds:
                obs, _ = test_env.reset(seed=args.seed)

        # Reduce dimensionality from 376D to 2D using UMAP
        umap_2D = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
        X_umap_2D = umap_2D.fit_transform(X)

        # Perform clustering using HDBSCAN
        hdb = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=10)
        labels_hdb = hdb.fit_predict(X_umap_2D)

        # Scatter plot of clusters
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(X_umap_2D[:, 0], X_umap_2D[:, 1], c=labels_hdb, cmap='Spectral', alpha=0.6, s=1)
        plt.colorbar(scatter, label="Cluster Label")
        plt.title("Cluster Visualization using UMAP + HDBSCAN")
        plt.xlabel("UMAP Component 1")
        plt.ylabel("UMAP Component 2")
        plt.show()

    finally:
        # Close the environments and tensorboard writer
        test_env.close()
