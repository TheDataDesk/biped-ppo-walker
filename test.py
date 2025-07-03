import gym
env = gym.make("BipedalWalker-v3")
env.reset()
env.render()
env.close()

print(env.observation_space)
print(env.action_space)
