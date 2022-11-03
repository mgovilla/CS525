#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import count
import random
from typing import Tuple
import numpy as np
from collections import defaultdict, deque
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN, self).__init__(env)
        self.device = "cuda"
        if args.test_dqn:
            # you can load your model here
            print('loading trained model')
            self.trained = True
            self.policy_net = torch.load(f="trained_policy_001.pth")
            self.policy_net.eval()
            return


        # create the nn model
        self.policy_net = DQN(*env.get_observation_space().shape,
                              env.get_action_space().n, device=self.device)
        self.policy_net.to(self.device)
        self.target_net = DQN(*env.get_observation_space().shape,
                              env.get_action_space().n, device=self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.to(self.device)
        self.target_net.eval()
        self.trained = False

        # create replay buffer
        self.buffer = deque(maxlen=10000)

        # constants
        self.UPDATE_TARGET_FREQ = 50
        self.gamma = 0.9
        self.iterations = 500
        self.TRAIN_STEPS = 2000
        
        # epsilon
        self.epsilon = 1.0
        self.decay_rate = (self.epsilon - 0.025) / self.TRAIN_STEPS


    def decay_epsilon(self):
        self.epsilon -= max(self.decay_rate, 0.025)

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        pass

    def make_action_test(self, observation): 
        with torch.no_grad():
            # get max reward action
            actions = self.policy_net(torch.tensor(np.array([observation.transpose()]), device=self.device, dtype=torch.float))
            return actions.max(1)[1].view(1, 1).item()

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        if test:
            return self.make_action_test(observation)

        # epsilon greedy
        p = random.random()
        if p > self.epsilon:
            with torch.no_grad():
                # get max reward action
                actions = self.policy_net(torch.tensor(np.array([observation.transpose()]), device=self.device, dtype=torch.float))
                return actions.max(1)[1].view(1, 1)

        return torch.tensor([[random.randrange(self.env.get_action_space().n)]], device=self.device, dtype=torch.long)

    def push(self, sars: Tuple):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.

        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        self.buffer.append(sars)

    def replay_buffer(self, batch_size):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """

        return random.sample(self.buffer, batch_size)

    def train(self):
        """
        start with policy_net = target_net
        for iterations:
            play a step in the game using the policy net
            record the step into the buffer

            sample from the buffer 
            compute the reward based on the target_net

            update the policy_net using a loss function (single step) on the reward
            every C iterations, set target_net = policy_net
        """

        optimizer = optim.RMSprop(self.policy_net.parameters())
        episode_rewards = []
        for n in range(self.iterations):
            state = self.env.reset()
            total_reward = 0
            # play an episode (until termination)
            done = False
            while not done:
                # decay epsilon 
                self.decay_epsilon()
                # pick an action
                action = self.make_action(state, False)

                # play a step in the game based on the policy net
                next_state, reward, done, _, _ = self.env.step(action.item())
                total_reward += reward
                # record (s, a, r, s')
                self.push((state, action.item(), reward,
                          next_state if not done else None))

                # sample from the buffer
                s, a, r, s_p = self.replay_buffer(self.batch_size)

                # calculate the error of the existing policy to the target
                # existing output
                existing_out = self.policy_net(torch.tensor(np.array([s.transpose()]), device=self.device, dtype=torch.float))

                y = r * torch.tensor(np.ones((1, self.env.get_action_space().n)), device=self.device, dtype=torch.float)
                if s_p is not None:
                    # add discounted future reward 
                    y = r + self.gamma*self.target_net(torch.tensor(np.array([s_p.transpose()]), device=self.device, dtype=torch.float))

                # Compute Huber loss
                criterion = torch.nn.SmoothL1Loss()
                loss = criterion(existing_out, y)

                optimizer.zero_grad()
                loss.backward()
                for param in self.policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

            episode_rewards.append(total_reward)

            if n % self.UPDATE_TARGET_FREQ == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        print(episode_rewards)
        torch.save(self.policy_net, "trained_policy_001.pth")
