from agent_dqn import Agent_DQN
from environment import Environment

env = Environment('BreakoutNoFrameskip-v4', None, atari_wrapper=True, test=False)
agent = Agent_DQN(env, None)

print()