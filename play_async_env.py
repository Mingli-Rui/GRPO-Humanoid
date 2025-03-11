import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv


def make_env():
    """Creates a new environment instance (must be picklable for multiprocessing)."""
    def _thunk():
        return gym.make("Humanoid-v4")
    return _thunk


if __name__ == "__main__":
    num_envs = 4
    envs = AsyncVectorEnv([make_env() for _ in range(num_envs)])  # Parallel execution

    obs, _ = envs.reset()
    done_flags = np.zeros(num_envs, dtype=bool)  # Track finished environments

    for _ in range(100):
        # ✅ Use idle actions (zero action) for finished environments
        actions = np.array([
            envs.single_action_space.sample() if not done_flags[i] else np.zeros(envs.single_action_space.shape)
            for i in range(num_envs)
        ])

        obs, rewards, dones, truncateds, infos = envs.step(actions)

        done_flags |= dones  # Mark finished environments
        print(done_flags)
        # ✅ Reset all environments together once all are done
        if done_flags.all():
            obs, _ = envs.reset()
            done_flags[:] = False  # Reset flags

    envs.close()