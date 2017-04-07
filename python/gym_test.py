import gym
from gym import envs
env = gym.make('FrozenLake-v0')

print(envs.registry.all())
print("action space")
print(env.action_space)
print("observation space")
print(env.observation_space)

for i_episode in range(5):
    observation = env.reset()
    for t in range(20):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
