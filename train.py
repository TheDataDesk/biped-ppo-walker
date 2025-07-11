import gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

# Create folders
os.makedirs("walker-models", exist_ok=True)

# Create vectorized environment
env = make_vec_env("BipedalWalker-v3", n_envs=1)

# Callback to print physics once at the beginning and once at the end
class PhysicsPrintOnceCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.printed = False

    def _on_step(self) -> bool:
        if not self.printed:
            raw_env = self.training_env.envs[0].unwrapped
            hull = raw_env.hull
            legs = raw_env.legs
            left_contact = any(getattr(legs[i], "ground_contact", False) for i in [0, 2])
            right_contact = any(getattr(legs[i], "ground_contact", False) for i in [1, 3])

            print("\nInitial Kinematic State")
            print(f"Hull Position: {hull.position}")
            print(f"Hull Velocity: {hull.linearVelocity}")
            print(f"Angular Velocity: {hull.angularVelocity}")
            print(f"Left Leg Contact: {1 if left_contact else 0}")
            print(f"Right Leg Contact: {1 if right_contact else 0}\n")

            self.printed = True
        return True

    def _on_training_end(self) -> None:
        raw_env = self.training_env.envs[0].unwrapped
        hull = raw_env.hull
        print("\nFinal Kinematic State")
        print(f"Hull Position: {hull.position}")
        print(f"Hull Velocity: {hull.linearVelocity}")
        print(f"Angular Velocity: {hull.angularVelocity}\n")

# Initialize PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model with physics info once at start and end
model.learn(total_timesteps=1_000_000, callback=PhysicsPrintOnceCallback())

# Save the model
model.save("walker-models/ppo-bipedalwalker")
env.close()
