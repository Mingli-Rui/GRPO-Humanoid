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

from lib.GRPOAgent import GRPOAgent
from lib.GRPOBuffer2 import GRPOBuffer2

def log_video(env, agent, device, video_path, seed, fps=30):
    """
    Log a video of one episode of the agent playing in the environment.
    :param env: a test environment which supports video recording and doesn't conflict with the other environments.
    :param agent: the agent to record.
    :param device: the device to run the agent on.
    :param video_path: the path to save the video.
    :param fps: the frames per second of the video.
    """
    # try 5 times, save the longest one
    longest_frames = []
    frames_lens = []
    max_try = 1
    for _ in range(max_try):
        frames = []
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            # Render the frame
            frames.append(env.render())
            # Sample an action
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(
                    torch.tensor(np.array([obs], dtype=np.float32), device=device))
            # Step the environment
            obs, _, terminated, _, _ = env.step(action.squeeze(0).cpu().numpy())
            done = terminated
        frames_lens.append(len(frames))
        if len(frames) > len(longest_frames):
            longest_frames = frames
    frames = longest_frames
    print('trajectories length:', frames_lens)

    # Save the video
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()


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
    videos_dir = os.path.join(current_dir, "videos", folder_name)
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
        for epoch in range(args.number):
            log_video(test_env, agent, device, os.path.join(videos_dir, f"render_{epoch}.mp4"), args.seed)

    finally:
        # Close the environments and tensorboard writer
        test_env.close()
