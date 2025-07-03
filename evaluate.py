from stable_baselines3 import PPO
import gym
import time

# Load environment and model
env = gym.make("BipedalWalker-v3")
model = PPO.load("ppo-bipedalwalker")

obs = env.reset()
done = False

while True:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(1.0 / 60.0)  # control the rendering speed
    if done:
        obs = env.reset()
