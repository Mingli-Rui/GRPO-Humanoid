import gymnasium as gym

# env = gym.make("LunarLander-v2", render_mode="human")
env = gym.make("Humanoid-v4", render_mode="human", max_episode_steps=10)
observation, info = env.reset()

episode_over = False
iteration = 0
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    iteration += 1
    print(f"iteration:{iteration}, action: {action}, observation: {observation.shape}, reward: {reward}, terminated: {terminated}, truncated: {truncated}")

    episode_over = terminated or truncated

print(f'total iteration: {iteration}')
env.close()
