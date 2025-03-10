import gymnasium as gym
import time

# env = gym.make("LunarLander-v2", render_mode="human")
env = gym.make("Humanoid-v4", render_mode="human", max_episode_steps=100)
observation, info = env.reset()

episode_over = False
iteration = 0
while iteration < 200:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    iteration += 1
    time.sleep(0.01)
    episode_over = terminated or truncated
    print(f"iteration:{iteration}, action: {action.shape}, observation: {observation.shape}, reward: {reward}, terminated: {terminated}, truncated: {truncated}, over: {episode_over}")

print(f'total iteration: {iteration}')
env.close()

