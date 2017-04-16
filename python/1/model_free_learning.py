### Episode model free learning using Q-learning and SARSA

# Do not change the arguments and output types of any of the functions provided! You may debug in Main and elsewhere.

import numpy as np
import gym
import time
from lake_envs import *

def learn_Q_QLearning(env, num_episodes=2000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99):
  """Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy.
  Update Q at the end of every episode.

  Parameters
  ----------
  env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
  num_episodes: int
    Number of episodes of training.
  gamma: float
    Discount factor. Number in range [0, 1)
  learning_rate: float
    Learning rate. Number in range [0, 1)
  e: float
    Epsilon value used in the epsilon-greedy method.
  decay_rate: float
    Rate at which learning rate falls. Number in range [0, 1)

  Returns
  -------
  np.array
    An array of shape [env.nS x env.nA] representing state, action values
  """

  ############################
  # YOUR IMPLEMENTATION HERE #
  ############################
  Q = np.zeros((env.nS,env.nA))
  #P[state][action] is tuples with (probability, nextstate, reward, terminal)
  P = env.P
  for ep in range(num_episodes):
      terminal = False
      s = 0
      while not terminal:
          #sample an action
          u = np.random.rand(1)
          if u > e:
              a = np.argmax(Q[s,:])
          else:
              a = np.random.randint(env.nA)

          #sample new state
          u = np.random.rand(1)
          for tup in P[s][a]:
              u = u - tup[0]
              if u <= 0:
                  t = tup
                  break

          q = t[2] + gamma*np.amax(Q[t[1],:])
          Q[s,a] = (1-lr)*Q[s,a] + lr*q
          s = t[1]
          if t[3]:
              terminal=True
      lr = lr*decay_rate
  return Q

def learn_Q_SARSA(env, num_episodes=2000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99):
  """Learn state-action values using the SARSA algorithm with epsilon-greedy exploration strategy
  Update Q at the end of every episode.

  Parameters
  ----------
  env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
  num_episodes: int
    Number of episodes of training.
  gamma: float
    Discount factor. Number in range [0, 1)
  learning_rate: float
    Learning rate. Number in range [0, 1)
  e: float
    Epsilon value used in the epsilon-greedy method.
  decay_rate: float
    Rate at which learning rate falls. Number in range [0, 1)

  Returns
  -------
  np.array
    An array of shape [env.nS x env.nA] representing state-action values
  """

  ############################
  # YOUR IMPLEMENTATION HERE #
  ############################

  return np.ones((env.nS, env.nA))

def render_single_Q(env, Q):
  """Renders Q function once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play Q function on. Must have nS, nA, and P as
      attributes.
    Q: np.array of shape [env.nS x env.nA]
      state-action values.
  """

  episode_reward = 0
  state = env.reset()
  done = False
  while not done:
    env.render()
    time.sleep(0.5) # Seconds between frames. Modify as you wish.
    action = np.argmax(Q[state])
    state, reward, done, _ = env.step(action)
    episode_reward += reward

  print "Episode reward: %f" % episode_reward

# Feel free to run your own debug code in main!
def main():
  env = gym.make('Stochastic-4x4-FrozenLake-v0')
  Q = learn_Q_QLearning(env)
  # Q = learn_Q_SARSA(env)
  render_single_Q(env, Q)

if __name__ == '__main__':
    main()
