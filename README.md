# PPO-Humanoid

This repository contains the implementation of a Proximal Policy Optimization (PPO) agent to control a humanoid in the
OpenAI Gymnasium Mujoco environment. The agent is trained to master complex humanoid locomotion using deep reinforcement
learning.

---

## Results

(insert gif here)

Here is a demonstration of the agent's performance after training for X epochs on the Humanoid-v4 environment.

---

## Installation

To get started with this project, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/ProfessorNova/PPO-Humanoid.git
    cd PPO-Humanoid
    ```

2. **Set Up Python Environment**:
   Make sure you have Python installed (tested with Python 3.10.11).

3. **Install Dependencies**:
   Run the following command to install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

   For proper PyTorch installation, visit [pytorch.org](https://pytorch.org/get-started/locally/) and follow the
   instructions based on your system configuration.

4. **Install Gymnasium Mujoco**:
   You need to install the Mujoco environment to simulate the humanoid:
    ```bash
    pip install gymnasium[mujoco]
    ```

5. **Train the Model**:
   To start training the model, run:
    ```bash
    python train.py --run-name "my_run"
    ```
   To train using a GPU, add the `--cuda` flag:
    ```bash
    python train.py --run-name "my_run" --cuda
    ```

6. **Monitor Training Progress**:
   You can monitor the training progress by viewing the videos in the `videos` folder or by looking at the graphs in
   TensorBoard:
    ```bash
    tensorboard --logdir "logs"
    ```

---

## Description

### Overview

This project implements a reinforcement learning agent using the Proximal Policy Optimization (PPO) algorithm, a popular
method for continuous control tasks. The agent is designed to learn how to control a humanoid robot in a simulated
environment.

### Key Components

- **Agent**: The core neural network model that outputs both policy (action probabilities) and value estimates.
- **Environment**: The Humanoid-v4 environment from the Gymnasium Mujoco suite, which provides a realistic physics
  simulation for testing control algorithms.
- **Buffer**: A class for storing trajectories (observations, actions, rewards, etc.) that the agent collects during
  interaction with the environment. This data is later used to calculate advantages and train the model.
- **Training Script**: The `train.py` script handles the training loop, including collecting data, updating the model,
  and logging results.

---

## Usage

### Training

You can customize the training by modifying the command-line arguments:

- `--n-envs`: Number of environments to run in parallel (default: 48).
- `--n-epochs`: Number of epochs to train the model (default: 5000).
- `--n-steps`: Number of steps per environment per epoch (default: 1024).
- `--batch-size`: Batch size for training (default: 8192).
- `--train-iters`: Number of training iterations per epoch (default: 20).

For example:

```bash
python train.py --run-name "experiment_1" --n-envs 64 --batch-size 4096 --train-iters 30 --cuda
```

All hyperparameters can be viewed either with `python train.py --help` or by looking at the
parse_args() function in `train.py`.
