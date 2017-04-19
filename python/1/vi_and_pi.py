### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters

import numpy as np
import gym
from lake_envs import *
import time
np.set_printoptions(precision=3)


def value_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
	"""
	Learn value function and policy by using value iteration method for a given
	gamma and environment.

	Parameters:
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""
	V = np.zeros(nS)
	V_old = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)
	############################
	# YOUR IMPLEMENTATION HERE #
	############################
	for i in range(max_iteration):
		for s in range(nS):
			best_v = 0
			for a in range(len(P[s])):
				v = 0
				for tup in P[s][a]:
					v += tup[0]*(tup[2]+gamma*V[tup[1]])
				if v > best_v:
					best_v = v
			V[s] = best_v
		policy = policy_improvement(P, nS, nA, V, policy, gamma)
		if np.amax(np.abs(V-V_old)) < tol:
			return V,policy
		V_old = V.copy()
		print 'not converged'
	return V, policy


def policy_evaluation(P, nS, nA, policy, gamma=0.9, max_iteration=1000, tol=1e-3):
	"""Evaluate the value function from a given policy.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	policy: np.array
		The policy to evaluate. Maps states to actions.
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns
	-------
	value function: np.ndarray
		The value function from the given policy.
	"""
	############################
	# YOUR IMPLEMENTATION HERE #
	############################
	val = np.zeros(nS)
	val_old = np.zeros(nS)
	for i in range(max_iteration):
		for s in range(nS):
			a = policy[s]
			val[s] = 0
			for tup in P[s][a]:
				val[s] += tup[0]*(tup[2]+gamma*val[tup[1]])

		if np.amax(np.abs(val-val_old)) < tol:
			return val
		val_old = val.copy()
	return val

def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
	"""Given the value function from policy improve the policy.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

	Returns
	-------
	new policy: np.ndarray
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""
	############################
	# YOUR IMPLEMENTATION HERE #
	############################
	pol = np.zeros(nS, dtype='int')
	for s in range(nS):
		best_v = 0
		best_a = 0
		for a in range(len(P[s])):
			v = 0
			for tup in P[s][a]:
				v += tup[0]*(tup[2]+gamma*value_from_policy[tup[1]])
			if v > best_v:
				best_v = v
				best_a = a
		pol[s] = best_a
	return pol

def policy_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
	"""Runs policy iteration.

	You should use the policy_evaluation and policy_improvement methods to
	implement this method.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""
	V = np.zeros(nS)
	V_old = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)
	############################
	# YOUR IMPLEMENTATION HERE #
	############################
	for i in range(max_iteration):
		print 'iteration {}'.format(i)
		policy = policy_improvement(P, nS, nA, V, policy, gamma)
		V = policy_evaluation(P, nS, nA, policy, gamma, max_iteration=1000, tol=1e-3)
		if np.amax(np.abs(V-V_old)) < tol:
			return V,policy
		V_old = V.copy()
	return V, policy



def example(env):
	"""Show an example of gym
	Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
	"""
	env.seed(0);
	from gym.spaces import prng; prng.seed(10) # for print the location
	# Generate the episode
	ob = env.reset()
	for t in range(100):
		env.render()
		a = env.action_space.sample()
		ob, rew, done, _ = env.step(a)
		if done:
			break
	assert done
	env.render();

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
	ob = env.reset()
	for t in range(100):
		env.render()
		time.sleep(0.5) # Seconds between frames. Modify as you wish.
		a = policy[ob]
		ob, rew, done, _ = env.step(a)
		episode_reward += rew
		if done:
			break
	assert done
	env.render();
	print "Episode reward: %f" % episode_reward

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
# Feel free to run your own debug code in main!
# Play around with these hyperparameters.
if __name__ == "__main__":
	#env = gym.make("Deterministic-4x4-FrozenLake-v0")
	env = gym.make("Stochastic-4x4-FrozenLake-v0")

	print env.__doc__
	print "Here is an example of state, action, reward, and next state"
	example(env)
	V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=100, tol=1e-3)
	V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=100, tol=1e-3)
	print MCRewardCalc(env,p_vi,100)
	print MCRewardCalc(env,p_pi,100)
	# render_single(env, p_vi)
	# render_single(env, p_pi)
