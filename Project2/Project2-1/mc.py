#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from unittest import result
import numpy as np
import random
from collections import defaultdict
from gym import Env
#-------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.
    You could test the correctness of your code
    by typing 'nosetests -v mc_test.py' in the terminal.

    You don't have to follow the comments to write your code. They are provided
    as hints in case you need.
'''
#-------------------------------------------------------------------------

def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and his otherwise

    Parameters:
    -----------
    observation

    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    # get parameters from observation
    ############################
    
    score, dealer_score, usable_ace = observation

    return int(score < 20)

def mc_prediction(policy, env: Env, n_episodes: int, gamma = 1.0):
    """Given policy using sampling to calculate the value function
        by using Monte Carlo first visit algorithm.

    Parameters:
    -----------
    policy: function
        A function that maps an obversation to action probabilities
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value

    Note: at the begining of each episode, you need initialize the environment using env.reset()
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # a nested dictionary that maps state -> value
    V = defaultdict(float)
    
    # generating episodes
    for _ in range(n_episodes):
        observation = env.reset()
        episode, terminated = [], False
        while not terminated:
            action = policy(observation)
            result, reward, terminated, truncated, info = env.step(action)
            episode.append((observation, action, reward))
            observation = result
        
        G = 0
        for t in range(len(episode)-1, -1, -1):
            G = gamma*G + episode[t][2] 

            if is_first_visit(episode, t):
                state = episode[t][0]
                returns_count[state] += 1
                returns_sum[state] += G

                V[state] = returns_sum[state] / returns_count[state]

    return V

def is_first_visit(episode, t):
    return all(map(lambda e: e[0]!=episode[t][0], episode[:t]))

def is_first_visit_Q(episode, t):
    return all(map(lambda e: e[0]!=episode[t][0] and e[1]!=episode[t][1], episode[:t]))

def epsilon_greedy(Q, state, nA, epsilon = 0.1):
    """Selects epsilon-greedy action for supplied state.

    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1

    Returns:
    --------
    action: int
        action based current state
    Hints:
    ------
    With probability (1 − epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """
    if random.random() > epsilon:
        # normal action
        return max(range(nA), key=lambda a: Q[state][a])
    
    return random.randint(0, nA-1)

def mc_control_epsilon_greedy(env, n_episodes, gamma = 1.0, epsilon = 0.1):
    """Monte Carlo control with exploring starts.
        Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-(0.1/n_episodes) during each episode
    and episode must > 0.
    """

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

     # generating episodes
    for t in range(n_episodes):
        epsilon -= 1 / n_episodes
        observation = env.reset()
        episode, terminated = [], False
        while not terminated:
            action = epsilon_greedy(Q, observation, env.action_space.n, epsilon)
            result, reward, terminated, truncated, info = env.step(action)
            episode.append((observation, action, reward))
            observation = result
        
        G = 0
        for t in range(len(episode)-1, -1, -1):
            G = gamma*G + episode[t][2] 

            if is_first_visit_Q(episode, t):
                state, action, _ = episode[t]
                returns_count[(state, action)] += 1
                returns_sum[(state, action)] += G

                Q[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]

    return Q
