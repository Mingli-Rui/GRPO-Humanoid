import argparse
import datetime
import os
import random
import time

import cv2
import gymnasium as gym
import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from lib.GRPOAgent import GRPOAgent
from lib.GRPOBuffer import GRPOBuffer
from lib.BufferGroup import BufferGroup


def log_video(env, agent, device, video_path, fps=30):
    """
    Log a video of one episode of the agent playing in the environment.
    :param env: a test environment which supports video recording and doesn't conflict with the other environments.
    :param agent: the agent to record.
    :param device: the device to run the agent on.
    :param video_path: the path to save the video.
    :param fps: the frames per second of the video.
    """
    frames = []
    obs, _ = env.reset()
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
    parser.add_argument("--cuda", default=False, action='store_true', help="Enable CUDA")
    parser.add_argument("--env", default="Humanoid-v4", help="Environment to use")
    parser.add_argument("--n-envs", type=int, default=48, help="Number of environments")
    parser.add_argument("--n-epochs", type=int, default=3000, help="Number of epochs to run")
    parser.add_argument("--n-steps", type=int, default=1024, help="Number of steps per epoch per environment")
    parser.add_argument("--batch-size", type=int, default=8192, help="Batch size")
    parser.add_argument("--train-iters", type=int, default=20, help="Number of training iterations")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.98, help="Lambda for GAE")
    parser.add_argument("--clip-ratio", type=float, default=0.1, help="PPO clip ratio")
    parser.add_argument("--ent-coef", type=float, default=1e-5, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=1.0, help="Value function coefficient")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--learning-rate-decay", type=float, default=0.999, help="Multiply with lr every epoch")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--reward-scale", type=float, default=0.005, help="Reward scaling")
    parser.add_argument("--render-epoch", type=int, default=50, help="Render every n-th epoch")
    parser.add_argument("--save-epoch", type=int, default=200, help="Save the model every n-th epoch")
    parser.add_argument("--group-size", type=int, default=10, help="Group size for GRPO")
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


# def reset_state(envs, num_envs, saved_states):
#     for i in range(num_envs):
#         env = envs.envs[i].unwrapped  # Unwrap to get raw MuJoCo env
#         mj_data = env.data  # MuJoCo state data
#
#         # Set qpos (joint positions) and qvel (joint velocities)
#         mj_data.qpos[:] = saved_states[i].qpos
#         mj_data.qvel[:] = saved_states[i].qvel
#
#         # Ensure physics is updated
#         env.model.step()  # Updates model state
#
#         # Perform a forward pass to update simulation
#         env.mujoco_renderer.sim_forward()
#
#         # Ensure state is applied correctly
#         env.data = mj_data

def seed_everything(seed):
    # random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # env.seed(seed)

if __name__ == "__main__":
    args = parse_args()
    # Support Mac mps
    device_name = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    # TODO use cpu for simplicity.
    # if device_name != "cuda" and torch.backends.mps.is_available():
    #     device_name = 'mps'
    print(f'train on device: {device_name}')
    device = torch.device(device_name)

    # Create the folders for logging
    current_dir = os.path.dirname(__file__)
    folder_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.run_name}"
    videos_dir = os.path.join(current_dir, "videos", folder_name)
    os.makedirs(videos_dir, exist_ok=True)
    checkpoint_dir = os.path.join(current_dir, "checkpoints", folder_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create the tensorboard writer
    log_dir = os.path.join(current_dir, "logs", folder_name)
    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Create the environments
    seed_env = make_env(args.env, reward_scaling=args.reward_scale, render=False)
    random.seed(args.seed)
    envs = gym.vector.SyncVectorEnv(
        [lambda: make_env(args.env, reward_scaling=args.reward_scale) for _ in range(args.n_envs)])
    test_env = make_env(args.env, reward_scaling=args.reward_scale, render=True)
    obs_dim = envs.single_observation_space.shape
    act_dim = envs.single_action_space.shape

    # Create the agent and optimizer
    agent = GRPOAgent(obs_dim[0], act_dim[0]).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.learning_rate_decay)
    print(agent.actor_mu)
    print(agent.actor_logstd)
    # print(agent.critic)

    # Create the buffer
    buffer = GRPOBuffer(obs_dim, act_dim, args.n_steps, args.n_envs, device, args.gamma, args.gae_lambda)
    assert args.batch_size % args.n_envs == 0
    batch_chunk_num = args.batch_size // args.n_envs
    buffer_group = BufferGroup(obs_dim, act_dim, batch_chunk_num, args.n_envs, device)

    # Start the training
    global_step_idx = 0
    start_time = time.time()
    next_obs = torch.tensor(np.array(envs.reset()[0], dtype=np.float32), device=device)
    next_terminateds = torch.tensor([float(False)] * args.n_envs, device=device)
    next_truncateds = torch.tensor([float(False)] * args.n_envs, device=device)

    reward_list = []
    shortest_traj = 0
    longest_traj = args.n_steps

    try:
        for epoch in range(1, args.n_epochs + 1):
            for _ in range(batch_chunk_num):
                # Get starting state
                max_skip_step_num = min(shortest_traj, args.n_steps)
                skip_step_num = random.randint(0, max_skip_step_num)
                shortest_traj = 0
                longest_traj = 0
                seed_everything(args.seed)
                # Reset and take some steps in the seed environment
                obs, _ = seed_env.reset(seed=args.seed)
                # if skip_step_num > 0:
                #     print(f'skip {skip_step_num} for training')
                pre_rewards = 0.0
                for skip_step in range(skip_step_num):  # Explore 10 steps
                    with torch.no_grad():
                        obs = torch.tensor(np.array(obs, dtype=np.float32), device=device)
                        action, _, _, _ = agent.get_action_and_value(obs)
                    obs, reward, done, truncated, _ = seed_env.step(action.cpu().numpy())
                    pre_rewards += reward
                    if done or truncated:
                        print(f"meet: skip_step {skip_step}")
                        pre_rewards = 0.0
                        obs, _ = seed_env.reset(seed=args.seed)  # Reset if episode ends

                # Save the final state from the seed environment
                saved_state = {
                    "qpos": seed_env.unwrapped.data.qpos.copy(),
                    "qvel": seed_env.unwrapped.data.qvel.copy()
                }

                # Reset envs to initialize memory
                obs, _ = envs.reset(seed=args.seed)

                for i in range(args.n_envs):
                    env_instance = envs.envs[i].unwrapped  # Access unwrapped MuJoCo env
                    mj_data = env_instance.data  # Get MuJoCo data structure

                    # Apply saved state
                    mj_data.qpos[:] = saved_state["qpos"]
                    mj_data.qvel[:] = saved_state["qvel"]

                    # Ensure physics updates (Corrected)
                    mujoco.mj_forward(env_instance.model, mj_data)  # Correct function

                # Collect trajectories
                not_done = torch.tensor(np.ones((1, args.n_envs), dtype=np.float32))
                for step_idx in range(0, args.n_steps):
                    global_step_idx += args.n_envs
                    obs = next_obs
                    terminateds = next_terminateds
                    truncateds = next_truncateds

                    # Sample the actions
                    with torch.no_grad():
                        actions, logprobs, _, values = agent.get_action_and_value(obs)
                        # values = values.flatten()

                    # Step the environment
                    next_obs, rewards, next_terminateds, next_truncateds, _ = envs.step(actions.cpu().numpy())

                    # # TODO: debugging
                    # if next_terminateds.sum() > 0.5 or next_truncateds.sum() > 0.5:
                    #     # some trajectory are truncated or terminated.
                    #     print(f'step: {step_idx}, {next_terminateds.sum()}, {next_truncateds.sum()}')
                    # parse everything to tensors
                    next_obs = torch.tensor(np.array(next_obs, dtype=np.float32), device=device)
                    reward_list.extend(rewards)
                    rewards = torch.tensor(rewards, dtype=torch.float32, device=device).view(-1)
                    next_terminateds = torch.tensor([float(term) for term in next_terminateds], device=device)
                    next_truncateds = torch.tensor([float(trunc) for trunc in next_truncateds], device=device)

                    # Store the step in the buffer
                    buffer.store(obs, actions, rewards, values, terminateds, truncateds, logprobs)
                    not_done = not_done * (1 - next_terminateds) * (1 - next_truncateds)
                    # print(f'step: {step_idx}, {shortest_traj} - {(args.n_envs - 1e-10)} / {longest_traj} {not_done.sum()} {not_done}')
                    if not_done.sum() >= (args.n_envs - 1e-10):
                        shortest_traj = step_idx
                    elif not_done.sum() <= 1e-10:
                        # longest_traj = torch.argmax(next_truncateds)
                        longest_traj = step_idx
                        # print(f'early-terminated after {step_idx} steps')
                        # epsilon = 0.5
                        # if random.random() > epsilon:
                        #     new_start_state = buffer.get_candidate_state(longest, step_idx).detach().numpy()
                        # else:
                        #     new_start_state = envs.reset()[0]
                        # reset_state(envs, args.n_envs, new_start_state)
                        buffer.early_terminate()
                        break

                # After the trajectories are collected, calculate the advantages and returns
                with torch.no_grad():
                    # Finish the last step of the buffer with the value of the last state
                    # and the terminated and truncated flags
                    # next_values = agent.get_value(next_obs).reshape(1, -1)
                    next_terminateds = next_terminateds.reshape(1, -1)
                    next_truncateds = next_truncateds.reshape(1, -1)
                    # traj_adv, traj_ret = buffer.calculate_grpo_advantages(next_terminateds, next_truncateds)
                    buffer_group.add(buffer, next_terminateds, next_truncateds)

            # Get one batch
            traj_adv, traj_ret, traj_obs, traj_act, traj_logprob = buffer_group.get()

            # Get the stored trajectories from the buffer
            # traj_obs, traj_act, traj_val, traj_logprob = buffer.get()

            # TODO: get first state only for training.
            # traj_obs, traj_act, traj_val, traj_logprob = traj_obs[0], traj_act[0], traj_val[0], traj_logprob[0]
            # Flatten the trajectories
            traj_obs = traj_obs.view(-1, *obs_dim)
            traj_act = traj_act.view(-1, *act_dim)
            traj_logprob = traj_logprob.view(-1)
            traj_adv = traj_adv.view(-1)
            traj_ret = traj_ret.view(-1)
            # traj_val = traj_val.view(-1)

            # TODO: only have one state not n_steps
            # Create an array of indices to sample from the trajectories
            # traj_indices = np.arange(args.n_steps * args.n_envs)
            traj_indices = np.arange(args.batch_size)

            sum_loss_policy = 0.0
            sum_loss_value = 0.0
            sum_entropy = 0.0
            sum_loss_total = 0.0
            for _ in range(args.train_iters):
                # Shuffle the indices
                np.random.shuffle(traj_indices)

                # Iterate over the batches
                # TODO: Only one record, don't need this loop
                # for start_idx in range(0, args.n_steps, args.batch_size):
                for start_idx in range(0, 1):
                    end_idx = start_idx + args.batch_size
                    batch_indices = traj_indices[start_idx:end_idx]

                    # Get the log probabilities, entropies and values
                    _, new_logprobs, entropies, new_values = agent.get_action_and_value(traj_obs[batch_indices],
                                                                                        traj_act[batch_indices])
                    ratios = torch.exp(new_logprobs - traj_logprob[batch_indices])

                    # normalize the advantages
                    batch_adv = traj_adv[batch_indices]
                    # TODO advantage is already normalized
                    # batch_adv = (batch_adv - batch_adv.mean()) / torch.max(batch_adv.std(),
                    #                                                        torch.tensor(1e-5, device=device))

                    # Calculate the policy loss
                    policy_loss1 = -batch_adv * ratios
                    policy_loss2 = -batch_adv * torch.clamp(ratios, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
                    policy_loss = torch.max(policy_loss1, policy_loss2).mean()
                    # # Calculate the policy loss
                    # policy_loss1 = batch_adv * ratios
                    # policy_loss2 = batch_adv * torch.clamp(ratios, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
                    # policy_loss = torch.min(policy_loss1, policy_loss2).mean()


                    # # Calculate the value loss
                    # new_values = new_values.view(-1)
                    # value_loss = 0.5 * ((new_values - traj_ret[batch_indices]) ** 2).mean()

                    # Calculate the entropy loss
                    entropy = entropies.mean()

                    # Calculate the total loss
                    # loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy
                    loss = policy_loss - args.ent_coef * entropy

                    # Optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                    sum_loss_policy += policy_loss.item()
                    # sum_loss_value += value_loss.item()
                    sum_entropy += entropy.item()
                    sum_loss_total += loss.item()

            # Update the learning rate
            scheduler.step()

            # Log info on console
            avg_reward = sum(reward_list) / len(reward_list)
            # Rescale the rewards
            avg_reward /= args.reward_scale
            print(f"Epoch {epoch} done in {time.time() - start_time:.2f}s. "
                  f"Skip/Shortest/Longest/Max: {skip_step_num} / {shortest_traj} / {longest_traj} {args.n_steps}. "
                  f"Avg reward: {(avg_reward + pre_rewards):.2f}. ")  # TODO: Need to revise pre_rewards. skipped reward is shared with all arms.
            reward_list = []

            # Every n epochs, log the video
            if epoch % args.render_epoch == 0:
                log_video(test_env, agent, device, os.path.join(videos_dir, f"epoch_{epoch}.mp4"))

            # Every n epochs, save the model
            if epoch % args.save_epoch == 0:
                torch.save(agent.state_dict(), os.path.join(checkpoint_dir, f"checkpoint_{epoch}.dat"))

            # Log everything to tensorboard
            writer.add_scalar("losses/policy_loss", sum_loss_policy / args.train_iters, global_step_idx)
            writer.add_scalar("losses/value_loss", sum_loss_value / args.train_iters, global_step_idx)
            writer.add_scalar("losses/entropy", sum_entropy / args.train_iters, global_step_idx)
            writer.add_scalar("losses/total_loss", sum_loss_total / args.train_iters, global_step_idx)
            writer.add_scalar("charts/avg_reward", avg_reward, global_step_idx)
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step_idx)
            writer.add_scalar("charts/SPS", global_step_idx / (time.time() - start_time), global_step_idx)

    finally:
        # Close the environments and tensorboard writer
        envs.close()
        seed_env.close()
        test_env.close()
        writer.close()

        # Save the model
        torch.save(agent.state_dict(), os.path.join(checkpoint_dir, "model.dat"))
