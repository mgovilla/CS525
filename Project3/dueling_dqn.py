#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch

class DQN(nn.Module):
    """Initialize a deep Q-learning network
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
        self.device = device

        # Network defined by the Deepmind paper

        # Convolutions on the frames on the screen
        # [(W−K+2P)/S]+1
        self.layer1 = nn.Conv2d(in_channels, 32, 8, 4)
        conv1_w = (in_size_w - 8) // 4 + 1
        conv1_h = (in_size_h - 8) // 4 + 1
        
        self.layer2 = nn.Conv2d(32, 64, 4, 2)
        conv2_w = (conv1_w - 4) // 2 + 1
        conv2_h = (conv1_h - 4) // 2 + 1

        self.layer3 = nn.Conv2d(64, 64, 3, 1)
        conv3_w = (conv2_w - 3) + 1
        conv3_h = (conv2_h - 3) + 1

        self.layer4 = nn.Flatten()

        self.layer5 = nn.Linear(64 * conv3_w * conv3_h, 512)

        # before the action layer, split the output into a value and advantage stream
        self.value = nn.Linear(512, 1)
        self.action = nn.Linear(512, num_actions)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = x.to(self.device)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x) # flatten

        x = F.relu(self.layer5(x))

        # combine the value and action layers
        v = F.relu(self.value(x))
        a = F.relu(self.action(x))
        Q = v + (a - a.mean(dim=1, keepdim=True))
        return Q
        