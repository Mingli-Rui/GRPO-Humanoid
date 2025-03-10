import torch
from sklearn.cluster import KMeans
import numpy as np
import hdbscan
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class GRPOBuffer2:
    """
    Buffer for storing trajectories
    """

    def __init__(self, obs_dim, act_dim, size, num_envs, device, gamma=0.99, gae_lambda=0.95):
        # Initialize buffer
        self.capacity = size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_buf = torch.zeros((size, num_envs, *obs_dim), dtype=torch.float32, device=device)
        self.act_buf = torch.zeros((size, num_envs, *act_dim), dtype=torch.float32, device=device)
        self.rew_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.val_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.term_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.trunc_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.logprob_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.ptr = 0
        self.cluster_size = 64
        self.min_cluster_size = 3

    def store(self, obs, act, rew, val, term, trunc, logprob):
        """
        Store one step of the trajectory. The step is in the first dimension of the tensors.
        In the second dimension are the different environments.
        """
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        # self.val_buf[self.ptr] = val
        self.term_buf[self.ptr] = term
        self.trunc_buf[self.ptr] = trunc
        self.logprob_buf[self.ptr] = logprob
        self.ptr += 1

    def get_similarity_advantage(self, observations):
        orig_shape = observations.shape
        observations_array = observations.view(-1, *self.obs_dim)

        # Convert to a single dataset (49,152 × num_features)
        states = observations_array.detach().numpy()
        num_steps = orig_shape[0]

        # Standardize data (HDBSCAN performs better on normalized data)
        scaler = StandardScaler()
        states_scaled = scaler.fit_transform(states)

        # Reduce dimensionality using PCA (helps HDBSCAN perform better)
        pca_dim = 10
        pca = PCA(n_components=pca_dim)  # Reduce to 2D for visualization
        states_pca = pca.fit_transform(states_scaled)

        states_pca = states_pca.reshape(orig_shape[0], orig_shape[1], pca_dim)

        # ==============================
        # Step 2: Compute Weighted Averaging for Smoothed States
        # ==============================
        def weighted_average(states_at_t, sigma=0.5):
            """
            Computes a weighted average of states at a given timestep across all trajectories.
            """
            num_trajectories = states_at_t.shape[0]
            query_state = np.mean(states_at_t, axis=0)  # Use mean state as the reference
            distances = np.linalg.norm(states_at_t - query_state, axis=1)

            # Compute Gaussian similarity-based weights
            weights = np.exp(-distances ** 2 / (2 * sigma ** 2))
            weights /= (np.sum(weights) + 1e-10)  # Normalize
            res = np.dot(weights, states_at_t)

            return res  # Compute weighted average state

        # Compute smoothed states (averaged over all trajectories per timestep)
        smoothed_states = np.array([weighted_average(states_pca[t]) for t in range(num_steps)])

        # ==============================
        # Step 3: Compute Similarity-Based Advantages
        # ==============================
        def compute_similarity_advantages(states, smoothed_states, sigma=0.5):
            """
            Computes similarity-based advantages by measuring the distance from smoothed states.
            """
            smoothed_states_expanded = np.expand_dims(smoothed_states, axis=1)
            distances = np.linalg.norm(states - smoothed_states_expanded, axis=2)  # Now shape = (1024, 48)
            advantages = np.exp(-distances ** 2 / (2 * sigma ** 2))  # Compute Gaussian similarity scores

            # Normalize advantages to have zero mean and unit variance
            return (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Compute similarity-weighted advantages
        advantages = compute_similarity_advantages(states_pca, smoothed_states)

        # # Normalize advantages to zero mean and unit variance
        # advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return advantages

    def get_cluster_assignments(self, observations, cluster_num):
        # Create Faiss KMeans model
        orig_shape = observations.shape
        observations_array = observations.view(-1, *self.obs_dim)
        kmeans = KMeans(n_clusters=cluster_num, random_state=1, algorithm="elkan", n_init="auto", max_iter=100)
        labels2 = kmeans.fit_predict(observations_array.detach().numpy())
        labels2 = labels2.reshape(orig_shape[:-1])

        # Convert to a single dataset (49,152 × num_features)
        states = observations_array.detach().numpy()

        # Standardize data (HDBSCAN performs better on normalized data)
        scaler = StandardScaler()
        states_scaled = scaler.fit_transform(states)

        # Reduce dimensionality using PCA (helps HDBSCAN perform better)
        pca = PCA(n_components=2)  # Reduce to 2D for visualization
        states_pca = pca.fit_transform(states_scaled)

        # Run HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=2)
        # clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=10)
        labels = clusterer.fit_predict(states_pca)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        effective_count = (labels > -1).sum()
        noise_count = (labels == -1).sum()
        print(f"num_clusters: {num_clusters}, count + noise = total: {effective_count} + {noise_count} = {labels.shape[0]}")

        # # Visualization of Clusters
        # plt.figure(figsize=(10, 6))
        # scatter = plt.scatter(states_pca[:, 0], states_pca[:, 1], c=labels, cmap='viridis', alpha=0.5, s=5)
        # plt.colorbar(scatter, label="Cluster Label")
        # plt.title("HDBSCAN Clustering of OpenAI Gym Humanoid States")
        # plt.xlabel("PCA Component 1")
        # plt.ylabel("PCA Component 2")
        # plt.show()
        # centroids = kmeans.cluster_centers_
        return labels, num_clusters

    def calculate_advantage2(self, returns, observations):
        """
        Calculates the advantage for each of the observations
        Args:
            returns: np.array of shape [batch size]
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            advantages: np.array of shape [batch size]
        """
        with torch.no_grad():
            cluster_num = observations.shape[0] // self.cluster_size
            print('cluster count:', cluster_num)
            labels, cluster_num = self.get_cluster_assignments(observations, cluster_num)
            new_advantages = np.zeros_like(returns, dtype=np.float32)
            for id in range(cluster_num):
                cluster_ids = labels == id
                # reward_count = cluster_ids.sum()
                cluster_reward = returns[labels == id]
                # print(f'{id}, {reward_count}, {cluster_reward.mean()}, {cluster_reward.std()}')
                this_advantage = (cluster_reward - cluster_reward.mean()) / (cluster_reward.std() + np.finfo(float).eps)
                copy_idx = 0
                for target_idx in range(observations.shape[0]):
                    if cluster_ids[target_idx]:
                        new_advantages[target_idx] = this_advantage[copy_idx]
                        copy_idx += 1
                # print(f'cluster: {id} , count: {reward_count}')

            return new_advantages

    def calculate_advantages(self, last_terminateds, last_truncateds):
        """
        Calculate advantages using Generalized Advantage Estimation (GAE).
        Should be called before get() to get the advantages and returns.
        :param last_vals: a tensor of shape (num_envs,) containing the value of the last state.
        :param last_terminateds: a tensor of shape (num_envs,) containing whether the last state was terminated.
        :param last_truncateds: a tensor of shape (num_envs,) containing whether the last state was truncated.
        :return: adv_buf, ret_buf
        """
        # Check if the Buffer is full
        assert self.ptr == self.capacity, "Buffer not full"
        last_vals = torch.zeros_like(self.val_buf[0])
        # Calculate the advantages
        with torch.no_grad():
            adv_buf = torch.zeros_like(self.rew_buf)
            # ret_buf = torch.zeros_like(self.rew_buf)
            last_gae = 0.0
            last_ret = 0.0
            for t in reversed(range(self.capacity)):
                # If the next state is the last state, use the last values and terminated flags
                next_vals = last_vals if t == self.capacity - 1 else self.val_buf[t + 1]
                term_mask = 1.0 - last_terminateds if t == self.capacity - 1 else 1.0 - self.term_buf[t + 1]
                trunc_mask = 1.0 - last_truncateds if t == self.capacity - 1 else 1.0 - self.trunc_buf[t + 1]

                # Only if the next state is marked as terminated the next value is 0
                # If the next state is marked as truncated, we still use the next value from the buffer
                # Last gae starts again from delta if the next state is terminated or truncated
                delta = self.rew_buf[t] + self.gamma * next_vals * term_mask - self.val_buf[t]
                last_gae = delta + self.gamma * self.gae_lambda * term_mask * trunc_mask * last_gae
                adv_buf[t] = last_gae

                # last_ret = self.rew_buf[t] + self.gamma * last_ret
                # ret_buf[t] = last_ret
            ret_buf = adv_buf  # + self.val_buf

            adv_array = self.get_similarity_advantage(self.obs_buf)
            adv_buf = torch.tensor(adv_array)
            # print(skipped_clusters)
            return adv_buf, ret_buf

    def get(self):
        """
        Call this at the end of the sampling to get the stored trajectories needed for training.
        :return: obs_buf, act_buf, val_buf, logprob_buf
        """
        assert self.ptr == self.capacity
        self.ptr = 0
        return self.obs_buf, self.act_buf, self.val_buf, self.logprob_buf
