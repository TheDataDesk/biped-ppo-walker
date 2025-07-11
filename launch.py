import gym

# Create environment
env = gym.make("BipedalWalker-v3", render_mode="human")
obs, _ = env.reset()

# Run random agent for a few steps
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
