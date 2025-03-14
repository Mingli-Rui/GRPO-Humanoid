import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv

def make_env():
    return lambda: gym.make("Humanoid-v4")

num_envs = 4
envs = SyncVectorEnv([make_env() for _ in range(num_envs)])  # Synchronous execution

obs, _ = envs.reset()
done_flags = np.zeros(num_envs, dtype=bool)  # Track finished environments

for _ in range(100):
    actions = np.array([envs.single_action_space.sample() if not done_flags[i] else np.zeros(envs.single_action_space.shape)
                        for i in range(num_envs)])  # ✅ Idle action for terminated envs

    obs, rewards, dones, truncateds, infos = envs.step(actions)  # Step all envs together

    print(done_flags)

    done_flags |= dones  # Mark environments that finished

    # ✅ Force sync: Reset all environments together only when all are done
    if done_flags.all():
        obs, _ = envs.reset()
        done_flags[:] = False  # Reset flags

envs.close()