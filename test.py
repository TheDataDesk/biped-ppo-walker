import gym
from stable_baselines3 import PPO

# Create environment
env = gym.make("BipedalWalker-v3", render_mode="human")

# Load model
model = PPO.load("/Users/sirishapadmasekhar/biped-ppo-walker/walker-models/ppo-bipedalwalker")

# Reset environment
obs, _ = env.reset()

# Run trained agent
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
    if done:
        obs, _ = env.reset()

env.close()