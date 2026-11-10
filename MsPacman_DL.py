#!/usr/bin/env python
# coding: utf-8

"""
Ms. Pac-Man DQN — Training and Evaluation Script
------------------------------------------------

This program trains a Deep Q-Network (DQN) agent to play the Atari game
Ms. Pac-Man using Gymnasium and PyTorch. The implementation follows the
classic DeepMind DQN architecture and training pipeline, including:

**Environment Setup**
   - Uses the ALE (Arcade Learning Environment) version of Ms. Pac-Man.
   - Applies standard Atari preprocessing:
       • Convert RGB frames to grayscale
       • Resize frames to 84×84
       • Stack the most recent 4 frames to capture motion
   - Enables sticky actions (`repeat_action_probability=0.25`) so the
     environment matches the stochastic testing conditions recommended
     in modern Atari benchmarks.

**Neural Network (MsPacmanDQN)**
   - A convolutional network with three conv layers followed by two fully
     connected layers.
   - Inputs: 4 stacked grayscale frames (shape 4×84×84)
   - Outputs: a Q-value for each legal action.
   - Matches the architecture from the original DQN Atari Ms PacMan paper DQN.

**Replay Memory**
   - Stores experience tuples (state, action, reward, next_state, done).
   - Enables experience replay, which stabilizes learning by breaking the
     correlation between consecutive samples.

**DQN Agent Logic**
   - ε-greedy exploration (ε decays from 1.0 → 0.1).
   - Q-learning updates computed from minibatches sampled from replay.
   - Maintains both:
       • Online network  — trains every step
       • Target network  — updated periodically for stable training
   - Uses Adam optimizer and Mean Squared Error (MSE) loss.

**Training Loop**
   - Interacts with the environment for a specified number of episodes.
   - Stores experiences, performs gradient updates once replay memory is
     sufficiently full.
   - Tracks episode return, episode length, loss, and ε.
   - Saves model checkpoints when the last-100-episode average reward
     reaches a new maximum.

**Testing / Evaluation**
   - Loads the best-performing checkpoint (`qnet_best.pt`).
   - Runs purely greedy (ε = 0) evaluation episodes.
   - Uses the same preprocessing and sticky-action setup as during
     training to avoid train–test mismatch.
   - Logs results to CSV for later analysis.

"""


# -----------------------------
# Imports for PyTorch
# -----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


import numpy as np
from collections import deque
import random
import csv, os, time, datetime
from datetime import datetime
import cv2
from copy import deepcopy

# -----------------------------
# Imports for Gym Atari MsPacman
# -----------------------------
import gymnasium as gym
import ale_py
from gymnasium.wrappers import FrameStackObservation
# SLURM Commands:
    # sbatch run_pacman.sbatch
    # squeue -u mjserena
    # sinfo
timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
print(timestamp)
print("Python:", os.sys.version)
print("PyTorch:", torch.__version__)
print("Gymnasium:",gym.__version__)
print("===== SLURM INFO =====")
slurm_job_id = os.environ.get("SLURM_JOB_ID")
if slurm_job_id:
    print(f"Slurm Job ID: {slurm_job_id}")
else:
    print("SLURM_JOB_ID not set.")
    
print("Node:", os.environ.get("SLURMD_NODENAME"))
print("Node list:", os.environ.get("SLURM_JOB_NODELIST"))
print("Partition:", os.environ.get("SLURM_JOB_PARTITION"))
print("======================")

# Check available hardware
if torch.cuda.is_available():
    print(f'GPU is available, using GPU {torch.cuda.get_device_name(0)}')
    device = torch.device('cuda')
elif torch.mps.is_available():
    print(f'Apple Metal Performance Shaders framework is available, using {torch.device("mps")}')
    device = torch.device('mps')
else:
    print('Hardware acceleration not available, using CPU')
    device = torch.device('cpu')



class AtariPreprocess(gym.ObservationWrapper):
    def __init__(self, env, height=84, width=84):
        super().__init__(env)
        self.height = height
        self.width = width
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width), dtype=np.uint8
        )

    def observation(self, obs):
        # obs: (H, W, 3)
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)



# MsPacmanDQN is a neural network that maps states → Q-values.

# This network receives:
# a stack of 4 game frames (shape (4, 84, 84)),
# 4 is the number of stacked frames
# 84×84 is the downsampled grayscale image (standard DQN pre-processing)

# and outputs:
# a vector of Q-values, one for each possible action.

class MsPacmanDQN(nn.Module):
    """
    Deep Q-Network for Atari Ms. Pac-Man.

    This model follows the classic DQN convolutional architecture used by DeepMind:
    - Input: 4 stacked grayscale frames of size 84×84
    - Output: Q-values for each possible discrete action

    The network progressively extracts spatial features at multiple scales:
    - conv1: detects large patterns (corridors, ghosts, pellets)
    - conv2: detects medium-scale features (bends, junctions)
    - conv3: fine detail (local edges, fruit, Pac-Man shape)

    The final fully connected layers map the extracted representation
    into a vector of Q-values, one for each action.
    """

    def __init__(self, num_actions, in_channels=4):
        super().__init__()
        # Input: (batch, in_channels=4, 84, 84)

        # | Layer | Input → Output   | Kernel | Stride | Purpose                        |
        # | ----- | ---------------- | ------ | ------ | ------------------------------ |
        # | conv1 | 4 → 32 channels  | 8×8    | 4      | Capture large spatial patterns |
        # | conv2 | 32 → 64 channels | 4×4    | 2      | Detect medium patterns         |
        # | conv3 | 64 → 64 channels | 3×3    | 1      | Fine details                   |
        # These layers reduce the image spatial size from 84×84 → 7×7 (exactly as in DQN).

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)


        # ---------------------------------------------------------
        # Fully Connected Layers (DQN "head")
        # ---------------------------------------------------------
        # Flatten 64 feature maps of size 7×7 = 3136 units.
        #
        # fc1:
        #   - Hidden layer with 512 units
        #   - Provides a dense, high-level representation of the scene.
        #
        # out:
        #   - Final layer, size = number of actions in the environment
        #   - Each output neuron is Q(s, a) for one specific action.
        # ---------------------------------------------------------

        # 84x84 → after convs this will be 7x7 if you follow the usual Atari setup
        # After the conv layers, the tensor is flattened:
        # 64 channels * 7 * 7 = 3136 numbers
        # This adds a fully-connected hidden layer with 512 units
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.out = nn.Linear(512, num_actions)


    def forward(self, x):
        # x expected as uint8 in [0,255] if coming straight from env
        # Normalize pixels
        # Input coming from Atari is uint8 in [0,255]
        # Dividing by 255 turns them into [0,1]
        # Three convolutional layers with ReLU
        # Flatten
        # Fully connected layer (512 units)
        # Output layer - Produces one Q-value per action

        # Normalize raw pixel values from [0, 255] → [0, 1]
        x = x / 255.0

        # Apply convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten all feature maps into a single vector per sample
        x = x.view(x.size(0), -1)

        # Fully connected layers (dense representation → Q-values)
        x = F.relu(self.fc1(x))
        return self.out(x)  # Q-values for all actions



class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)



class GymDQNAgent:
    def __init__(
        self,
        qnet,
        tnet,
        num_actions,
        replay_memory,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=100000,
        batch_size=32,
        target_update_freq=1000,
        device="cpu",
        optimizer=None,
    ):
        self.qnet = qnet
        self.tnet = tnet
        self.num_actions = num_actions
        self.memory = replay_memory
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device

        self.optimizer = optimizer

        self.total_steps = 0

    def epsilon(self):
        # Epsilon Decay: 
        # Typically, epsilon is initialized to a relatively high value (e.g., 1.0) 
        # at the beginning of training, ensuring a high degree of exploration (random action, 
        # even if it doesn't currently appear to be the best.)
        # This allows the agent to gather a diverse range of experiences and build up 
        # its initial estimates of Q-values. 
        # As training progresses and the agent learns more about the environment, epsilon
        # epsilon is gradually decreased (linear decay). 
        # This shifts the balance towards exploitation, allowing the agent to leverage 
        # its learned knowledge to achieve higher rewards.

        # frac will be a based upon the number of steps in the last episode
        frac = min(1.0, (self.total_steps / self.epsilon_decay_steps))
        # Linearly reducing epsilon between start and end values
        new_eps = self.epsilon_start + (frac * (self.epsilon_end - self.epsilon_start))
        return new_eps

    def select_action(self, state):
        # state: torch tensor (1, C, H, W) or numpy array you convert here
        p = np.random.rand()       # random number in [0,1)
        if p < self.epsilon():            # explore with probability = epsilon
            # If exploring → return random integer
            # pick a random action
            random_action = np.random.randint(self.num_actions)
            return random_action

        # If exploiting → run Q-network under .eval()
        self.qnet.eval()

        # What are gradients?
        # A gradient = the derivative of the loss with respect to a weight.
        # It tells the network: “If I nudge this weight up or down, how much would my loss change?”
        # Gradients are used for learning, because optimizers like Adam or SGD use them to update weights
        # So gradients are only needed when we are training (doing backpropagation).
        
        # Do not track gradients for anything inside this block
        # During action selection: not updating weights.
        # Are only: running a forward pass, picking the max-Q action
        # Do not want PyTorch to compute gradients
  
        with torch.no_grad():
            q_values = self.qnet(state)  
            action = q_values.argmax(dim=1).item()
        return action

    def store_transition(self, s, a, r, s_next, done):
        # Make sure states are uint8 before storing
        s      = np.asarray(s,      dtype=np.uint8)
        s_next = np.asarray(s_next, dtype=np.uint8)
        self.memory.push(s, a, r, s_next, done)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None  # not enough samples
    
        # 1. Sample batch — returns list of tuples
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
    
        # 2. Convert to numpy arrays
        states      = np.stack(states)        # shape (B, 4, 84, 84)
        next_states = np.stack(next_states)   # shape (B, 4, 84, 84)
        actions     = np.array(actions, dtype=np.int64)
        rewards     = np.array(rewards, dtype=np.float32)
        dones       = np.array(dones, dtype=np.float32)
    
        # 3. Convert to tensors
        states      = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        actions     = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards     = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones       = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
    
        # 4. Q(s,a) using online network
        q_values = self.qnet(states)                  # (B, num_actions)
        q_sa = q_values.gather(1, actions)            # (B, 1)
    
        # 5. TD target using target network
        # TD stands for Temporal Difference.
        # What Is the TD Target? It is the value you want your Q-network 
        # to produce for the (state, action) pair you just experienced.        
        with torch.no_grad():
            next_q_values = self.tnet(next_states)    # (B, num_actions)
            max_next_q = next_q_values.max(dim=1, keepdim=True)[0]
            target = rewards + self.gamma * (1.0 - dones) * max_next_q
    
        # 6. Compute MSE (Mean Squared Error) loss
        # How wrong the Q-network’s prediction was on this sample.
        # We quantify that error as loss. 
        loss = torch.nn.functional.mse_loss(q_sa, target)
    
        # 7. Optimize
        # The optimizer (Adam) then back-propagates that loss and 
        # updates weights to make the network’s Q-value predictions 
        # closer to the Bellman target.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        # 8. Update target network
        self.total_steps += 1
        if self.total_steps % self.target_update_freq == 0:
            self.tnet.load_state_dict(self.qnet.state_dict())
    
        return loss.item()
    


# Create an environment for training
train_env = gym.make(
    "ALE/MsPacman-v5",
    frameskip=4,
    repeat_action_probability=0.25,  # <-- stochasticity ON
    render_mode=None
)

# (1) Per-frame preprocessing
train_env = AtariPreprocess(train_env)         # custom wrapper

# (2) Temporal stacking of 4 frames
train_env = FrameStackObservation(train_env, stack_size=4) # provided by Gymnasium
print("train_env:", train_env.observation_space, train_env.action_space)

num_actions = train_env.action_space.n

print('What device(s) can torch use in your environment?')

if torch.cuda.is_available():
    device = torch.device('cuda')
    print (f'GPU is available, using { device }')
elif torch.mps.is_available():
    device = torch.device('mps')
    print (f'Apple Metal Performance Shaders framework is available, using { device }')
else:
    device = torch.device('cpu')
    print (f'Hardware acceleration not available , using { device }')
print()

# qnet = “online network”
# The CNN that:
# sees every new state during training
# computes Q-values to pick actions
# gets updated every training step by gradient descent
# learns from replay buffer samples
# is constantly changing as it learns
qnet = MsPacmanDQN(num_actions, in_channels=4).to(device)

# print structure of CNN
print("CCN Model Structure:\n")
for name, layer in qnet.named_children():
    print(f"({name}): {layer}")

    for param_name, param in layer.named_parameters():
        print(f"    {param_name}: {param.size()}")
print()
model_stats = summary(
    qnet,
    input_size=(1, 4, 84, 84),
    device=device,
    col_names=("input_size", "output_size", "num_params", "kernel_size"),
    depth=2,
    verbose=0
)
with open("cnn_baseline_summary.txt", "w") as f:
    f.write(str(model_stats))
print(model_stats)

# tnet = "target network"
# is a copy of qnet, but NOT updated every step
# stays frozen for many iterations
# is used only to compute stable targets for learning
# is updated only occasionally (e.g., every 5,000 steps or every N episodes)
tnet = MsPacmanDQN(num_actions, in_channels=4).to(device)
tnet.load_state_dict(qnet.state_dict())

memory = ReplayMemory(capacity=20000)  
optimizer = torch.optim.Adam(qnet.parameters(), lr=1e-4)

agent = GymDQNAgent(
    qnet=qnet,
    tnet=tnet,
    num_actions=num_actions,
    replay_memory=memory,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay_steps=100000,
    batch_size=32,
    target_update_freq=1000,
    device=device,
    optimizer=optimizer,
)

num_episodes = 10000
min_replay_size = 5000

episode_returns = []
episode_lengths = []
episode_durations = []
best_last100_avg = -float("inf")
episode_losses = []
log = "log_pacman.csv"
# Set debugging off
print_debugging = False
timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
print(timestamp)
# Open log file
log_path = log    
if log_path:
    if os.path.exists(log_path):
        pass
    else:
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "phase", "episode", "return", 
                "steps", "avg_loss","epsilon"
            ])
        print(log_path, "created")


print("Running", num_episodes, "Training episodes")
phase = "train"
for episode in range(num_episodes):
    obs, info = train_env.reset()

    # obs is an LazyFrames shaped (4, 84, 84)
    state = np.array(obs, copy=False)  # you can normalize later in forward()

    done = False
    episode_return = 0
    ep_start_time = time.time()
    steps = 0

    while not done:

        steps += 1
        
        # 1. Select action
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action = agent.select_action(state_tensor)

        # 2. Step environment
        next_obs, reward, terminated, truncated, info = train_env.step(action)
        done = terminated or truncated
        next_state = np.array(next_obs, copy=False)

        # 3. Store transition
        agent.store_transition(state, action, reward, next_state, done)
        episode_return += reward
        state = next_state

        # 4. Train if we have enough data
        if len(memory) >= min_replay_size:
            # Save returned loss
            loss_value = agent.train_step()   
            # Store loss for monitoring
            if loss_value is not None:        
                episode_losses.append(loss_value)
            
    if len(episode_losses) > 0:
        avg_loss = sum(episode_losses) / len(episode_losses)
    else:
        avg_loss = None

    ep_duration = time.time() - ep_start_time

    # Save stats for summaries
    episode_returns.append(episode_return)
    episode_lengths.append(steps)
    episode_durations.append(ep_duration)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    if print_debugging:
        print(
            f"[{timestamp}] Episode {episode}, return: {episode_return}, steps: {steps}, avg_loss: {avg_loss}, epsilon: {agent.epsilon():.3f}"
    )

    # Append CSV row
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            phase,
            episode,
            episode_return,
            steps,
            avg_loss,
            agent.epsilon()])

    # Print summary every 100 episodes
    if (episode + 1) % 100 == 0:
        window = 100
        last_returns = episode_returns[-window:]
        last_lengths = episode_lengths[-window:]
        last_durations = episode_durations[-window:]

        avg_all = sum(episode_returns) / len(episode_returns)
        avg_last_ret = sum(last_returns) / len(last_returns)
        avg_last_len = sum(last_lengths) / len(last_lengths)
        avg_last_time = sum(last_durations) / len(last_durations)

        print("Reinforcement Learning Status:")
        print(f"\tCompleted {episode + 1} training episodes")
        print(f"\tAverage Rewards over all training: {avg_all:.2f}")
        print(f"\tAverage Rewards for last {window} episodes: {avg_last_ret:.2f}")
        print(f"\tAverage Length  for last {window} episodes: {avg_last_len:.2f}")
        print(f"\tAverage Time per episode (last {window}): {avg_last_time:.4f} seconds")
        print("-" * 60)

         # Save best-performing network so far
        if avg_last_ret > best_last100_avg:
            best_last100_avg = avg_last_ret
            torch.save(qnet.state_dict(), "qnet_best.pt")
            print(f"qnet saved with best last-100 avg reward = {avg_last_ret:.2f}")



# Now create an environment for testing (repeat_action_probability=0.25)
# Using ALE sticky actions is the standard, recommended, 
# and most academically accepted way to test Atari DQN performance.
test_env = gym.make(
    "ALE/MsPacman-v5",
    frameskip=4,
    repeat_action_probability=0.25,  # <-- stochasticity ON
    render_mode="human"
)

test_env = AtariPreprocess(test_env)
test_env = FrameStackObservation(test_env, stack_size=4)
if torch.cuda.is_available():
    device = "cuda"  
else:
    device = "cpu"

qnet.to(device)

def run_test_episode(qnet, env, device):
    # Reset environment and get the initial stacked frames
    obs, info = env.reset()
    state = np.array(obs, copy=False)   # Preprocessed observation (4 stacked frames)

    done = False
    episode_return = 0
    steps = 0

    # Put network in evaluation mode (disables dropout, batchnorm updates)
    qnet.eval()

    while not done:
        steps += 1

        # Convert state to a tensor on the correct device (1, 4, 84, 84)
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0)

        # Greedy action selection: no epsilon, no gradients (pure evaluation)
        with torch.no_grad():
            q_values = qnet(state_tensor)
            action = q_values.argmax(dim=1).item()

        # Step the environment using the chosen action
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Accumulate reward for episode score
        episode_return += reward

        # Update the state with the next stacked frames
        state = np.array(next_obs, copy=False)


    print(f"[Testing] return: {episode_return}, steps: {steps}")
    return episode_return, steps

if os.path.exists("qnet_best.pt"):
    print("loading qnet_best.pt")
    state_dict = torch.load("qnet_best.pt", map_location=device, weights_only=True)
    qnet.load_state_dict(state_dict)


num_episodes = 100
phase = "test"
print("Running", num_episodes, "Testing episodes")
for episode in range(num_episodes):
    episode_return, steps = run_test_episode(qnet, test_env, device)
    # Append CSV row
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            phase,
            episode,
            episode_return,
            steps,
            None,
            None])