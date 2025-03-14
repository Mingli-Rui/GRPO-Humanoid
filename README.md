# GRPO-Humanoid

This repository contains the implementation of a Group Relative Policy Optimization (GRPO) agent to control a humanoid in the
OpenAI Gymnasium Mujoco environment.

This repository is forked from https://github.com/ProfessorNova/PPO-Humanoid which is an implementation of PPO. And it is used as baseline.

---

## Results of PPO (baseline)
![PPO Demo Gif](https://github.com/ProfessorNova/PPO-Humanoid/blob/main/docs/demo.gif)

## Results of GRPO with KMean cluster
![GRPO with KMean Demo Gif](https://github.com/Mingli-Rui/GRPO-Humanoid/blob/m3/docs/demo_kmean.gif)

## Results of GRPO with Similarity
![GRPO with Similarity Demo Gif](https://github.com/Mingli-Rui/GRPO-Humanoid/blob/m3/docs/demo_sim.gif)

## Demos on Youtube
[![GRPO with KMean Demo Youtube](https://img.youtube.com/vi/pGOiNQByCzY/0.jpg)](https://www.youtube.com/watch?v=pGOiNQByCzY)
[![GRPO with Similarity Demo Youtube](https://img.youtube.com/vi/M1l95Pdd4BU/0.jpg)](https://www.youtube.com/watch?v=M1l95Pdd4BU)

Here is a demonstration of the agent's performance after training for 3000 epochs on the Humanoid-v4 environment.

---

## Installation

To get started with this project, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Mingli-Rui/GRPO-Humanoid.git
    cd GRPO-Humanoid
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
    # NOTE: You may need to quote the name on Mac
    pip install "gymnasium[mujoco]"
    ```

5. **Train the Model (Baseline)**:
   To start training the model, run:
    ```bash
    python train.py --run-name "my_ppo"
    ```
   To train using a GPU, add the `--cuda` flag:
    ```bash
    python train.py --run-name "my_ppo" --cuda
    ```

6. **Train a model with GRPO based on KMean clustering:**
   We implemented GRPO based on KMean.
    ```bash
    python train_grpo_kmean.py --run-name "run_kmean" --seed 1
    ```

7. **Train a model GRPO with weighted average reward based on similarity:**
   We implemented an algorithm to compute weighted average reward.
    ```bash
    python train_similarity.py --run-name "run_sim" --seed 1
    ```

8. **Monitor Training Progress**:
   You can monitor the training progress by viewing the videos in the `videos` folder or by looking at the graphs in
   TensorBoard:
    ```bash
    tensorboard --logdir "logs"
    ```

9. **Render videos for a model**:
   You can render videos from the trained model.
    ```bash
    python render.py --run-name 'render' --model checkpoints/2025-03-10_21-35-16_run_sim/model.dat --number 100 --seed 1
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
- `--n-epochs`: Number of epochs to train the model (default: 3000).
- `--n-steps`: Number of steps per environment per epoch (default: 1024).
- `--batch-size`: Batch size for training (default: 8192).
- `--train-iters`: Number of training iterations per epoch (default: 20).

For example:

```bash
python train.py --run-name "experiment_1" --n-envs 64 --batch-size 4096 --train-iters 30 --cuda
```

All hyperparameters can be viewed either with `python train.py --help` or by looking at the
parse_args() function in `train.py`.

---

## Performance

Here are the specifications of the system used for training:

- **CPU**: AMD Ryzen 9 5900X
- **GPU**: Nvidia RTX 3080 (12GB VRAM)
- **RAM**: 64GB DDR4
- **OS**: Windows 11

The training process took about 5 hours to complete 3000 epochs on the Humanoid-v4 environment.

### Hyperparameters

The hyperparameters used for training are as follows:

| param               | value       | 
|---------------------|-------------| 
| run_name            | baseline    | 
| cuda                | True        | 
| env                 | Humanoid-v4 |
| n_envs              | 48          |
| n_epochs            | 3000        |
| n_steps             | 1024        |
| batch_size          | 8192        | 
| train_iters         | 20          | 
| gamma               | 0.995       | 
| gae_lambda          | 0.98        |
| clip_ratio          | 0.1         | 
| ent_coef            | 1e-05       |
| vf_coef             | 1.0         |
| learning_rate       | 0.0003      | 
| learning_rate_decay | 0.999       |
| max_grad_norm       | 1.0         | 
| reward_scale        | 0.005       | 
| render_epoch        | 50          |
| save_epoch          | 200         |

### Statistics

### Performance Metrics:

The following charts provide insights into the performance during training:

- **Reward**:
  ![Reward](https://github.com/ProfessorNova/PPO-Humanoid/blob/main/docs/charts_avg_reward.svg)

  As seen in the chart, the agent's average reward is still increasing after 3000 epochs,
  indicating that the agent has not yet reached its full potential and could benefit from further training.

- **Policy Loss**:
  ![Policy Loss](https://github.com/ProfessorNova/PPO-Humanoid/blob/main/docs/losses_policy_loss.svg)

- **Value Loss**:
  ![Value Loss](https://github.com/ProfessorNova/PPO-Humanoid/blob/main/docs/losses_value_loss.svg)

  In the chart above, the value loss first increases and then decreases until it plateaus after 100M steps. This
  behavior is expected as the agent first explores the environment and then learns to predict the value of states more
  accurately.

- **Entropy Loss**:
  ![Entropy](https://github.com/ProfessorNova/PPO-Humanoid/blob/main/docs/losses_entropy.svg)
