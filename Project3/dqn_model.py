#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    def __init__(self, in_size_w, in_size_h, in_channels=4, num_actions=4, device="cpu"):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.

        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        KERNEL_SIZE=3
        self.device = device
        self.conv1 = nn.Conv2d(in_channels, 24, KERNEL_SIZE)
        self.bn1 = nn.BatchNorm2d(24)
        conv1_w = in_size_w - KERNEL_SIZE + 1
        conv1_h = in_size_h - KERNEL_SIZE + 1

        self.conv2 = nn.Conv2d(24, 64, KERNEL_SIZE)
        self.bn2 = nn.BatchNorm2d(64)
        conv2_w = conv1_w - KERNEL_SIZE + 1
        conv2_h = conv1_h - KERNEL_SIZE + 1

        self.head = nn.Linear(64*conv2_w*conv2_h, num_actions)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.head(x.view(x.size(0), -1))
