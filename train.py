from stable_baselines3 import PPO
import gym

# Create the environment
env = gym.make("BipedalWalker-v3")

# Define the model (policy, env)
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=100_000)

# Save the model
model.save("ppo-bipedalwalker")

env.close()
