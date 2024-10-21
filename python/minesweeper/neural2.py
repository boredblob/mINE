import math
import torch
from torch import nn, unsqueeze
from torch.nn import functional as F
import numpy as np
from collections import deque
import itertools
import random
from datetime import datetime
import csv

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from minesweeper import minesweeper_game

game_width = 30
game_height = 16
num_mines = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.flatten_size = game_width * game_height * 64
        
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, game_width*game_height*2)

    def forward(self, x):
        x = x.view(-1, 2, game_height, game_width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.flatten_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def act(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        q_values = self(state_t.unsqueeze(0))
        max_q = torch.argmax(q_values, dim=1)[0]
        return max_q.item()

class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
    
    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        priorities = np.array([abs(experience[2]) for experience in self.buffer])
        probabilities = priorities / np.sum(priorities)
        sample_indices = np.random.choice(range(len(self.buffer)), size=batch_size, p=probabilities)
        return [self.buffer[i] for i in sample_indices]

def get_state(game_instance: minesweeper_game):
    mines_grid = torch.asarray(game_instance.mines_grid, dtype=torch.float32) / 8
    cleared_grid = torch.asarray(game_instance.cleared_grid, dtype=torch.float32)
    uncleared_mask = (cleared_grid == 0) * -1.0
    masked_grid = torch.where(cleared_grid==1, mines_grid, uncleared_mask)
    flags_grid = torch.asarray(game_instance.flags_grid, dtype=torch.float32)

    masked_grid_2d = masked_grid.view(game_height, game_width)
    flags_grid_2d = flags_grid.view(game_height, game_width)

    state = torch.stack([masked_grid_2d, flags_grid_2d], dim=0).to(device)
    return state

def to_action(action):
    clamped_action = max(0, min(game_width*game_height*2-1, action))
    clear_or_flag = clamped_action // (game_width*game_height)
    idx = action - (clear_or_flag*game_width*game_height)
    return (clear_or_flag, idx)

def game_step(game_instance: minesweeper_game, action):
    num_clears_before = game_instance.num_cleared_cells
    num_flags_before = game_instance.num_flagged_cells

    clear_or_flag, idx = to_action(action)

    game_status = -1
    if clear_or_flag == 0:
        game_status = game_instance.clear_cell(*game_instance.to_coords(idx))
    if clear_or_flag == 1:
        game_status = game_instance.flag_cell(*game_instance.to_coords(idx))

    state = get_state(game_instance)
    num_clears_after = game_instance.num_cleared_cells
    num_flags_after = game_instance.num_flagged_cells

    done = False
    num_cells_cleared = num_clears_after - num_clears_before
    reward = 0
    if num_cells_cleared > 0:
        reward = min(0.5, math.sqrt(num_cells_cleared/(game_width*game_height)))
    if num_flags_after > num_mines:
        reward = -0.25
    if num_flags_after < num_flags_before:
        reward = -0.8
    if (num_clears_after == num_clears_before) and (num_flags_after == num_flags_before):
        reward = -1
    if game_status != 0:
        done = True
        reward = game_status

    reward = max(-1, min(1, reward))

    return (state, reward, done)

def decay_epsilon(eps_start, eps_end, eps_decay_rate, current_step):
    return eps_end + (eps_start - eps_end) * np.exp(-1 * eps_decay_rate * current_step)

# Training parameters
eps_count = 0
batch_size = 128
gamma = 0.95
epsilon_start = 1
epsilon_end = 0.001
epsilon_decay_rate = 0.0025

# Initialize networks
online_net = Network().to(device)
target_net = Network().to(device)
target_net.load_state_dict(online_net.state_dict())
optimiser = torch.optim.Adam(online_net.parameters())

# Initialize replay memory
replay_memory = ReplayBuffer(10000)  # Increased buffer size

# Initialize memory with random experiences
num_random_game_actions = 1000
game_instance = minesweeper_game(game_width, game_height, num_mines)
for i in range(num_random_game_actions):
    prev_state = get_state(game_instance)
    action = random.randint(0, game_width*game_height*2-1)
    state, reward, done = game_step(game_instance, action)
    experience = (prev_state, action, reward, done, state)
    replay_memory.add(experience)

    if done:
        game_instance = minesweeper_game(game_width, game_height, num_mines)

# Training loop
num_training_games = 500
steps_per_game = []
max_steps_per_game = 150

for t in range(num_training_games):
    game_instance = minesweeper_game(game_width, game_height, num_mines)
    prev_state = get_state(game_instance).to(device)
    won_game = False

    for step in itertools.count():
        eps_count += 1
        epsilon = decay_epsilon(epsilon_start, epsilon_end, epsilon_decay_rate, eps_count)
    
        # Select action
        action = random.randint(0, game_width*game_height*2-1)
        if random.random() > epsilon:
            with torch.no_grad():
                action = online_net.act(prev_state)

        # Take action and store experience
        state, reward, done = game_step(game_instance, action)
        if done and (reward == 1):
            won_game = True
        
        experience = (prev_state, action, reward, done, state)
        replay_memory.add(experience)
        prev_state = state

        # Sample batch and prepare tensors
        experiences = replay_memory.sample(batch_size)
        
        # Separate experiences into batches
        batch_states = torch.stack([torch.as_tensor(e[0], dtype=torch.float32) for e in experiences]).to(device)
        batch_actions = torch.tensor([e[1] for e in experiences], dtype=torch.int64).to(device)
        batch_rewards = torch.tensor([e[2] for e in experiences], dtype=torch.float32).to(device)
        batch_dones = torch.tensor([e[3] for e in experiences], dtype=torch.float32).to(device)
        batch_next_states = torch.stack([torch.as_tensor(e[4], dtype=torch.float32) for e in experiences]).to(device)

        # Get current Q values
        current_q_values = online_net(batch_states)
        current_q_values = current_q_values.gather(1, batch_actions.unsqueeze(1))

        # Get next Q values
        with torch.no_grad():
            max_next_q_values = target_net(batch_next_states).max(1)[0]
            expected_q_values = batch_rewards + (1 - batch_dones) * gamma * max_next_q_values
            expected_q_values = expected_q_values.unsqueeze(1)

        # Compute loss and optimize
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # Update target network
        if eps_count % num_training_games == 0:
            target_net.load_state_dict(online_net.state_dict())

        if done or step > max_steps_per_game:
            print(f"Game {t}: steps={step}, won={won_game}, loss={loss.item():.4f}")
            steps_per_game.append((t, step, won_game, loss.item()))
            break

# Save model and training statistics
datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
torch.save(online_net.state_dict(), f"C:/Users/omerk/Documents/GitHub/mINE/python/minesweeper/models/{datetime_str}.pth")

with open(f"C:/Users/omerk/Documents/GitHub/mINE/python/minesweeper/models/{datetime_str}_stats.csv", 'w', newline='') as tf:
    file_writer = csv.writer(tf)
    file_writer.writerow(['game', 'steps', 'won', 'loss'])
    for i in steps_per_game:
        file_writer.writerow(i)