import numpy as np
import torch
from lib.GRPOBuffer import GRPOBuffer


class BufferGroup:
    """
    Buffer for storing trajectories
    """

    def __init__(self, obs_dim, act_dim, size, num_envs, device):
        # Initialize buffer
        self.capacity = size
        self.obs_buf = torch.zeros((size, num_envs, *obs_dim), dtype=torch.float32, device=device)
        self.act_buf = torch.zeros((size, num_envs, *act_dim), dtype=torch.float32, device=device)
        self.ret_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.adv_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.logprob_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.ptr = 0

    def add(self, buffer, last_terminateds, last_truncateds):
        """
        Store one step of the trajectory. The step is in the first dimension of the tensors.
        In the second dimension are the different environments.
        """
        adv_buf, ret_buf = buffer.calculate_grpo_advantages(last_terminateds, last_truncateds)
        self.obs_buf[self.ptr] = buffer.obs_buf[0]
        self.act_buf[self.ptr] = buffer.act_buf[0]
        self.ret_buf[self.ptr] = ret_buf
        self.adv_buf[self.ptr] = adv_buf
        self.logprob_buf[self.ptr] = buffer.logprob_buf[0]
        self.ptr += 1
        buffer.reset()

    def get(self):
        """
        Call this at the end of the sampling to get the stored trajectories needed for training.
        :return: obs_buf, act_buf, val_buf, logprob_buf
        """
        assert self.ptr == self.capacity
        self.ptr = 0
        return self.adv_buf, self.ret_buf, self.obs_buf, self.act_buf, self.logprob_buf
