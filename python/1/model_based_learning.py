### Episodic Model Based Learning using Maximum Likelihood Estimate of the Environment

# Do not change the arguments and output types of any of the functions provided! You may debug in Main and elsewhere.

import numpy as np
import gym
import time
from lake_envs import *

from vi_and_pi import value_iteration

def initialize_P(nS, nA):
  """Initializes a uniformly random model of the environment with 0 rewards.

    Parameters
    ----------
    nS: int
      Number of states
    nA: int
      Number of actions

    Returns
    -------
    P: np.array of shape [nS x nA x nS x 4] where items are tuples representing transition information
      P[state][action] is a list of (prob, next_state, reward, done) tuples.
  """
  P = [[[(1.0/nS, i, 0, False) for i in range(nS)] for _ in range(nA)] for _ in range(nS)]

  return P

def initialize_counts(nS, nA):
  """Initializes a counts array.

    Parameters
    ----------
    nS: int
      Number of states
    nA: int
      Number of actions

    Returns
    -------
    counts: np.array of shape [nS x nA x nS]
      counts[state][action][next_state] is the number of times that doing "action" at state "state" transitioned to "next_state"
  """
  counts = [[[0 for _ in range(nS)] for _ in range(nA)] for _ in range(nS)]

  return counts

def initialize_rewards(nS, nA):
  """Initializes a rewards array. Values represent running averages.

    Parameters
    ----------
    nS: int
      Number of states
    nA: int
      Number of actions

    Returns
    -------
    rewards: array of shape [nS x nA x nS]
      counts[state][action][next_state] is the running average of rewards of doing "action" at "state" transtioned to "next_state"
  """
  rewards = [[[0 for _ in range (nS)] for _ in range(nA)] for _ in range(nS)]

  return rewards

def counts_and_rewards_to_P(counts, rewards):
  """Converts counts and rewards arrays to a P array consistent with the Gym environment data structure for a model of the environment.
    Use this function to convert your counts and rewards arrays to a P that you can use in value iteration.

    Parameters
    ----------
    counts: array of shape [nS x nA x nS]
      counts[state][action][next_state] is the number of times that doing "action" at state "state" transitioned to "next_state"
    rewards: array of shape [nS x nA x nS]
      counts[state][action][next_state] is the running average of rewards of doing "action" at "state" transtioned to "next_state"

    Returns
    -------
    P: np.array of shape [nS x nA x nS x 4] where items are tuples representing transition information
      P[state][action] is a list of (prob, next_state, reward, done) tuples.
  """
  nS = len(counts)
  nA = len(counts[0])
  P = [[[] for _ in range(nA)] for _ in range(nS)]

  for state in range(nS):
    for action in range(nA):
      if sum(counts[state][action]) != 0:
        for next_state in range(nS):
          if counts[state][action][next_state] != 0:
            prob = float(counts[state][action][next_state]) / float(sum(counts[state][action]))
            reward = rewards[state][action][next_state]
            P[state][action].append((prob, next_state, reward, False))
      else:
        prob = 1.0 / float(nS)
        for next_state in range(nS):
          P[state][action].append((prob, next_state, 0, False))

  return P

def update_mdp_model_with_history(counts, rewards, history):
  """Given a history of an entire episode, update the count and rewards arrays

    Parameters
    ----------
    counts: array of shape [nS x nA x nS]
      counts[state][action][next_state] is the number of times that doing "action" at state "state" transitioned to "next_state"
    rewards: array of shape [nS x nA x nS]
      counts[state][action][next_state] is the running average of rewards of doing "action" at "state" transtioned to "next_state"
    history:
      a list of [state, action, new_state, REWARD, done]
  """

  # HINT: For terminal states, we define that the probability of any action returning the state to itself is 1 (with zero reward)
  # Make sure you record this information in your counts array by updating the counts for this accordingly for your
  # value iteration to work.

  ############################
  # YOUR IMPLEMENTATION HERE #
  ############################
  for t in history:
      s,a,sprime,r,terminal = t
      a = int(a)

      if not terminal:
          counts[s][a][sprime] += 1
      else:
          counts[s][a][sprime] += 1
          counts[sprime][a][sprime]+=1
      rewards[s][a][sprime] += 1.0/counts[s][a][sprime]*(r-rewards[s][a][sprime])

  return counts, rewards

def learn_with_mdp_model(env, num_episodes=5000, gamma = 0.95, e = 0.8, decay_rate = 0.999):
  """Build a model of the environment and use value iteration to learn a policy. In the next episode, play with the new
    policy using epsilon-greedy exploration.

    Your model of the environment should be based on updating counts and rewards arrays. The counts array counts the number
    of times that "state" with "action" led to "next_state", and the rewards array is the running average of rewards for
    going from at "state" with "action" leading to "next_state".

    For a single episode, create a list called "history" with all the experience
    from that episode, then update the "counts" and "rewards" arrays using the function "update_mdp_model_with_history".

    You may then call the prewritten function "counts_and_rewards_to_P" to convert your counts and rewards arrays to
    an environment data structure P consistent with the Gym environment's one. You may then call on value_iteration(P, nS, nA)
    to get a policy.

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
    policy: np.array
      An array of shape [env.nS] representing the action to take at a given state.
    """

  P = initialize_P(env.nS, env.nA)
  counts = initialize_counts(env.nS, env.nA)
  rewards = initialize_rewards(env.nS, env.nA)

  ############################
  # YOUR IMPLEMENTATION HERE #
  ############################
  policy = np.zeros((env.nS))
  rs = []
  for ep in range(num_episodes):
      terminal = False
      s = 0
      history = []
      while not terminal:
          #sample an action
          u = np.random.rand(1)
          if u > e:
              a = policy[s]
          else:
              a = np.random.randint(env.nA)

          #sample new state
          u = np.random.rand(1)
          for tup in env.P[s][a]:
              u = u - tup[0]
              if u <= 0:
                  t = tup
                  break
          history.append((s,a,t[1],t[2],t[3]))

          s = t[1]
          if t[3]:
              terminal=True
              rs.append(t[2])

      counts,rewards = update_mdp_model_with_history(counts,rewards,history)
      P = counts_and_rewards_to_P(counts,rewards)
      value,policy = value_iteration(P, env.nS, env.nA, gamma, 200, 1e-3)
      e = e*decay_rate
  np.save('model_based_rewards.npy',rs)
  return policy

def render_single(env, policy):
  """Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
  """

  episode_reward = 0
  state = env.reset()
  done = False
  while not done:
    env.render()
    time.sleep(0.5) # Seconds between frames. Modify as you wish.
    action = policy[state]
    state, reward, done, _ = env.step(action)
    episode_reward += reward

  print "Episode reward: %f" % episode_reward

# Feel free to run your own debug code in main!
def MCRewardCalc(env, Q, num_episodes=100):
    total_reward = 0
    for i in range(num_episodes):
      episode_reward = 0
      state = env.reset()
      done = False
      while not done:
        #env.render()
        #time.sleep(0.5) # Seconds between frames. Modify as you wish.
        action = Q[state]
        state, reward, done, _ = env.step(action)
        episode_reward += reward
      print "{} Episode reward: {}".format(i,episode_reward)
      total_reward+=episode_reward
    return float(total_reward)/num_episodes

def main():
  env = gym.make('Stochastic-4x4-FrozenLake-v0')
  policy = learn_with_mdp_model(env)
  render_single(env, policy)
  print MCRewardCalc(env, policy)
if __name__ == '__main__':
    main()
