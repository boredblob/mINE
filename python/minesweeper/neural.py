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
    
    # Compute the size of the flattened output after convolutional layers
    # Assuming input is of size (input_channels, grid_size, grid_size)
    self.flatten_size = (game_width // 2) * (game_height // 2) * 64
    
    # Fully connected layers
    self.fc1 = nn.Linear(self.flatten_size, 128)
    self.fc2 = nn.Linear(128, game_width*game_height*2)

  def forward(self, x):
      # x is the input tensor representing the game grid (with 2 channels: mines_grid and flags_grid)
      # Reshape input to match convolutional input (batch_size, 2 channels, grid_size, grid_size)
      x = x.view(-1, 2, x.size(1), x.size(2))
      
      # Pass through convolutional layers
      x = F.relu(self.conv1(x))  # First conv layer + ReLU activation
      x = F.relu(self.conv2(x))  # Second conv layer + pooling
      
      # Flatten the output from the convolutional layers
      x = x.view(-1, self.flatten_size)
      
      # Pass through fully connected layers
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
    priorities = [abs(experience[2]) for experience in self.buffer]
    probabilities = priorities / np.sum(priorities)
    sample_indices = np.random.choice(range(len(self.buffer)), size=batch_size, p=probabilities)
    return [self.buffer[i] for i in sample_indices]


def get_state(game_instance: minesweeper_game):
  # info going into model: distances from mines, whether cells are cleared, 
  mines_grid = torch.asarray(game_instance.mines_grid, dtype=torch.float32) / 8
  cleared_grid = torch.asarray(game_instance.cleared_grid, dtype=torch.float32)
  uncleared_mask = (cleared_grid == 0) * -1.0
  masked_grid = torch.where(cleared_grid==1, mines_grid, uncleared_mask)
  flags_grid = torch.asarray(game_instance.flags_grid, dtype=torch.int64)

  masked_grid_2d = masked_grid.view(game_width, game_height)
  flags_grid_2d = flags_grid.view(game_width, game_height)

  state = torch.stack([masked_grid_2d, flags_grid_2d], dim=0)
  return state

def to_action(action):
  clamped_action = max(0, min(game_width*game_height*2-1, action))
  clear_or_flag = clamped_action // (game_width*game_height)
  idx = action - (clear_or_flag*game_width*game_height)
  return (clear_or_flag, idx)

def game_step(game_instance: minesweeper_game, action):
  # return calculate_state(), reward, game_finished?
  num_clears_before = game_instance.num_cleared_cells
  num_flags_before = game_instance.num_flagged_cells

  clear_or_flag, idx = to_action(action)

  game_status = -1 # lose by default if action is somehow invalid
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
  reward = 0.5 if num_clears_after > num_clears_before else 0 # reward for clearing cells
  if num_flags_after > num_mines: # penalise a bit for having too many flags
    reward = -0.25
  if num_flags_after < num_flags_before: # penalise for removing flags
    reward = -0.8
  if (num_clears_after == num_clears_before) and (num_flags_after == num_flags_before): # penalise for doing nothing
    reward = -1
  if game_status != 0:
    done = True
    reward = game_status

  reward = max(-1, min(1, reward)) # normalising reward

  return (state, reward, done)

def decay_epsilon(eps_start, eps_end, eps_decay_rate, current_step):
  return eps_end + (eps_start - eps_end) * np.exp(-1 * eps_decay_rate * current_step)

eps_count = 0
batch_size = 1024
gamma = 0.95

online_net = Network()
target_net = Network()
# online_net.load_state_dict(torch.load("C:/Users/omerk/Documents/GitHub/mINE/python/minesweeper/models/2024-10-2015-28-07", weights_only=True))
target_net.load_state_dict(online_net.state_dict())
optimiser = torch.optim.Adam(online_net.parameters())
online_net.to(device)
target_net.to(device)

epsilon_start = 1
epsilon_end = 0.001
epsilon_decay_rate = 0.0025

episode_durations = []

# setup memory with random experiences
replay_memory = ReplayBuffer(1000)
num_random_game_actions = 1000
game_instance = minesweeper_game(game_width, game_height, num_mines)
for i in range(num_random_game_actions):
  prev_state = get_state(game_instance).to(device)
  action = random.randint(0, game_width*game_height*2-1)
  state, reward, done = game_step(game_instance, action)
  experience = (prev_state, action, reward, done, state)
  replay_memory.add(experience)

  if done:
    game_instance = minesweeper_game(game_width, game_height, num_mines)

# run games for training
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
  
    action = random.randint(0, game_width*game_height*2-1)
    if random.random() > epsilon: 
      action = online_net.act(prev_state)

    state, reward, done = game_step(game_instance, action)
    if done and (reward == 1): won_game = True
    experience = (prev_state, action, reward, done, state)
    replay_memory.add(experience)
    prev_state = state

    experiences = replay_memory.sample(batch_size)
    states = torch.stack([e[0] for e in experiences]).to(device)
    actions = torch.asarray([e[1] for e in experiences]).to(device)
    rewards = torch.asarray([e[2] for e in experiences]).to(device)
    dones = torch.asarray([e[3] for e in experiences]).to(device)
    new_states = torch.stack([e[4] for e in experiences]).to(device)

    states_t = torch.as_tensor(states, dtype=torch.float32).to(device)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1).to(device)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(device)
    dones_t = torch.as_tensor(dones, dtype=torch.int64).unsqueeze(-1).to(device)
    new_states_t = torch.as_tensor(new_states, dtype=torch.float32).to(device)

    q_values = online_net(states_t)
    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t).to(device)
   
    target_q_output = target_net(new_states_t)
    target_q_values = target_q_output.max(dim=1, keepdim=True)[0].to(device)

    # print(dones_t.shape, target_q_values.shape)
    optimal_q_values = rewards_t + gamma * (1 - dones_t) * target_q_values
    print(optimal_q_values.shape)

    loss = nn.functional.smooth_l1_loss(action_q_values, optimal_q_values)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    if eps_count % num_training_games == 0:
      target_net.load_state_dict(online_net.state_dict())

    if done or step > max_steps_per_game:
      print(f"game {t} took {step} steps")
      steps_per_game.append((t, step, won_game))
      break

datetime_str = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
torch.save(online_net.state_dict(), f"C:/Users/omerk/Documents/GitHub/mINE/python/minesweeper/models/{datetime_str}")

with open(f"C:/Users/omerk/Documents/GitHub/mINE/python/minesweeper/models/{datetime_str}.csv", 'w', newline='') as tf:
  file_writer = csv.writer(tf)
  for i in steps_per_game:
    file_writer.writerow(i)