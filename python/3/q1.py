import math
import gym
from frozen_lake import *
import numpy as np
import time
from utils import *
runs = {}
# def sample_action(q):
# 	r = np.random.rand
# 	for a in q:
# 		r -= a
# 		if r <=0:
# 			return a
# 	return a
#
# def recompute_value(nSA,nSASP,R,Q,gamma):
# 	nS,nA = nSA.shape
# 	P = nSASP.astype(float)/nSA[:,:,np.newaxis]
#
# 	q = np.argmax(Q,axis=1)
# 	Q = R + gamma* P.dot(q)
# 	return Q.copy()
def rmax(env, gamma, m, R_max, epsilon, num_episodes, max_step = 6):
	"""Learn state-action values using the Rmax algorithm

	Args:
	----------
	env: gym.core.Environment
		Environment to compute Q function for. Must have nS, nA, and P as
		attributes.
	m: int
		Threshold of visitance
	R_max: float
		The estimated max reward that could be obtained in the game
	epsilon:
		accuracy paramter
	num_episodes: int
		Number of episodes of training.
	max_step: Int
		max number of steps in each episode

	Returns
	-------
	np.array
	  An array of shape [env.nS x env.nA] representing state-action values
	"""

	Q = np.ones((env.nS, env.nA)) * R_max / (1 - gamma)
	R = np.zeros((env.nS, env.nA))

	nSA = np.zeros((env.nS, env.nA))
	nSASP = np.zeros((env.nS, env.nA, env.nS))
	########################################################
	#                   YOUR CODE HERE                     #
	########################################################
	for e in range(num_episodes):
		total_reward = 0
		s = env.reset()
		for i in range(max_step):
			a = np.argmax(Q[s,:])
			st,r,done,_ = env.step(a)
			if nSA[s,a] < m:
				nSA[s,a] += 1
				nSASP[st,a,s] += 1
				R[s,a] += r
				if nSA[s,a] == m:
					for i in range(50):
						for sp in range(env.nS):
							for ap in range(env.nA):
								if nSA[sp,ap] >= m:
									Q[sp,ap] = R[sp,ap]/nSA[sp,ap]
									for spp in range(env.nS):
										Q[sp,ap] += gamma*nSASP[spp,ap,sp]/nSA[sp,ap]*np.amax(Q[spp,:])
			s = st
			total_reward+=r
		if runs.has_key(m):
			if len(runs[m])==0:
				runs[m].append(total_reward)
			else:
				avg = runs[m][-1] + 1.0/(len(runs[m])+1)*(total_reward-runs[m][-1])
				runs[m].append(avg)
	print 'nSA\n {}\n, nSASP\n {}\n, R\n {}\n, Q\n {}\n'.format(nSA,nSASP,R,Q)
	########################################################
	#                    END YOUR CODE                     #
	########################################################
	return Q

def learn_Q_QLearning(env, num_episodes=5000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99):
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
	#Q = np.zeros((env.nS,env.nA))
	Q = np.random.rand(env.nS,env.nA)
	#P[state][action] is tuples with (probability, nextstate, reward, terminal)
	P = env.P
	rewards = []
	for ep in range(num_episodes):
		total_reward = 0
		terminal = False
		s = 0
		#for i in range(6):
		for i in range(6):
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

		  total_reward += t[2]

		e = e*decay_rate
		if len(rewards)==0:
			rewards.append(total_reward)
		else:
			print total_reward
			avg = rewards[-1] + 1.0/(len(rewards)+1)*(total_reward-rewards[-1])
			rewards.append(avg)
	return Q.copy(),rewards

def main():

	env = FrozenLakeEnv(is_slippery=False)
	print env.__doc__
	Q = rmax(env, gamma = 0.99, m=10, R_max = 1, epsilon = 0.1, num_episodes = 1000)
	render_single_Q(env, Q)
	print Q

	for m in [5,10,20,50,100]:
		runs[m] = []
		q = rmax(env, gamma = 0.99, m=m, R_max = 1, epsilon = 0.1, num_episodes = 10000)

	q_,rewards = learn_Q_QLearning(env, num_episodes=10000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99)
	colors = ['red','green','blue','yellow','black']
	import matplotlib.pyplot as plt
	plt.figure()
	for i in range(len(runs.keys())):
		k = runs.keys()[i]
		c = colors[i]
		plt.plot(runs[k],linewidth=2,label='m={}'.format(k),color=c)
	plt.plot(rewards,linewidth=2,label='Q',color='teal')
	plt.xlabel('episode')
	plt.ylabel('average total reward')
	plt.legend(loc='bottom right')
	plt.savefig('m_Reward_q1.png',dpi=300)
	plt.close()
	print rewards
if __name__ == '__main__':
	main()
