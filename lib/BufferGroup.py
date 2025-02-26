import numpy as np
import torch
from lib.GRPOBuffer import GRPOBuffer


class BufferGroup:
    def __init__(self, group_size):
        self.group_size = group_size
        self.buffer_list = []
        self.advantages = None
        self.returns = None

    def add(self, buf):
        self.buffer_list.append(buf)

    def __calculate_advantages(self):
        with torch.no_grad():
            buffer_returns = [buf.get_returns() for buf in self.buffer_list]
            self.returns = torch.stack(buffer_returns, dim=0)
            self.advantages = (self.returns - self.returns.mean(dim=0)) / self.returns.std(dim=0)

    def get_advantage(self, idx):
        if self.advantages is None:
            self.__calculate_advantages()
        return self.buffer_list[idx], self.advantages[idx], self.returns[idx]