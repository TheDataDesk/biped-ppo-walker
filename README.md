# Bipedal Walker PPO Agent

This project implements a reinforcement learning agent trained to walk using **Proximal Policy Optimization (PPO)** in the `BipedalWalker-v3` environment from **OpenAI Gym**. The agent learns bipedal locomotion from scratch through interaction with a continuous control environment, using reward signals to guide policy updates.

## Objective

To train a simulated bipedal robot to walk autonomously using deep reinforcement learning — without imitation learning, predefined motion policies, or manual control logic.

---

## Overview

- **Environment:** OpenAI Gym `BipedalWalker-v3`
- **Algorithm:** PPO (Proximal Policy Optimization)
- **Framework:** [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) (PyTorch)
- **Observation Space:** 24 continuous inputs  
  - Includes hull angle/velocity, joint angles, ground contacts, lidar-based terrain sensing
- **Action Space:** 4-dimensional continuous vector (torques to hips and knees)
- **Training:** 1M+ timesteps using a reward function encouraging forward motion and penalizing inefficient torque usage

---

## Dependencies

```bash
pip install -r requirements.txt
```

## Training 

python train.py


## Testing 

To run a trained agent and visualize walking behavior:

python test.py

## Demo

[![Watch the demo](https://www.youtube.com/watch?v=n8mxB51R0Lk)

