import math
import torch
from torch import nn
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

# Start with a smaller grid size
INITIAL_WIDTH = 4
INITIAL_HEIGHT = 4
INITIAL_MINES = 4

# Will gradually increase to:
MAX_WIDTH = 30
MAX_HEIGHT = 16
MAX_MINES = 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DualHeadNetwork(nn.Module):
    def __init__(self, width, height):
        super().__init__()
        self.width = width
        self.height = height
        
        # Shared layers
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Action head
        self.action_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.action_fc1 = nn.Linear(32 * width * height, 256)
        self.action_fc2 = nn.Linear(256, width * height * 2)
        
        # Value head
        self.value_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.value_fc1 = nn.Linear(32 * width * height, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Shared layers
        x = x.view(-1, 2, self.height, self.width)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Action head
        action = F.relu(self.action_conv(x))
        action = action.view(-1, 32 * self.width * self.height)
        action = F.relu(self.action_fc1(action))
        action = self.action_fc2(action)
        
        # Value head
        value = F.relu(self.value_conv(x))
        value = value.view(-1, 32 * self.width * self.height)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return action, value
    
    def act(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        action_values, state_value = self(state_t.unsqueeze(0))
        return action_values[0].argmax().item(), state_value.item()

class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
    
    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        return random.sample(list(self.buffer), batch_size)

def get_state(game_instance: minesweeper_game):
    mines_grid = torch.asarray(game_instance.mines_grid, dtype=torch.float32) / 8
    cleared_grid = torch.asarray(game_instance.cleared_grid, dtype=torch.float32)
    uncleared_mask = (cleared_grid == 0) * -1.0
    masked_grid = torch.where(cleared_grid==1, mines_grid, uncleared_mask)
    flags_grid = torch.asarray(game_instance.flags_grid, dtype=torch.float32)

    masked_grid_2d = masked_grid.view(game_instance.grid_height, game_instance.grid_width)
    flags_grid_2d = flags_grid.view(game_instance.grid_height, game_instance.grid_width)

    state = torch.stack([masked_grid_2d, flags_grid_2d], dim=0).to(device)
    return state

def to_action(action, width, height):
    clamped_action = max(0, min(width*height*2-1, action))
    clear_or_flag = clamped_action // (width*height)
    idx = action - (clear_or_flag*width*height)
    return (clear_or_flag, idx)

def calculate_reward(prev_state, new_state, game_instance, action_type, won, lost):
    """More sophisticated reward calculation"""
    reward = 0
    
    # Base rewards/penalties for game outcomes
    if won:
        return 10.0
    if lost:
        return -5.0
    
    # Reward for revealing safe squares
    prev_cleared = prev_state[0].sum().item()
    new_cleared = new_state[0].sum().item()
    cells_cleared = new_cleared - prev_cleared
    if cells_cleared > 0:
        reward += min(8.0, cells_cleared * 0.5)  # Reward per cleared cell
    
    # Penalty for unnecessary flags
    if action_type == 1:  # Flag action
        if game_instance.num_flagged_cells > game_instance.num_mines:
            reward -= 0.3
    
    # Small step penalty to encourage efficiency
    reward -= 0.1
    
    return reward

def game_step(game_instance: minesweeper_game, action, prev_state):
    clear_or_flag, idx = to_action(action, game_instance.grid_width, game_instance.grid_height)
    coords = game_instance.to_coords(idx)
    
    game_status = 0
    if clear_or_flag == 0:
        game_status = game_instance.clear_cell(*coords)
    else:
        game_status = game_instance.flag_cell(*coords)

    new_state = get_state(game_instance)
    won = game_status == 1
    lost = game_status == -1
    done = won or lost
    
    reward = calculate_reward(prev_state, new_state, game_instance, clear_or_flag, won, lost)
    
    return new_state, reward, done

def train(curriculum_level=0):
    # Set grid size based on curriculum level
    width = min(INITIAL_WIDTH + curriculum_level * 2, MAX_WIDTH)
    height = min(INITIAL_HEIGHT + curriculum_level * 1, MAX_HEIGHT)
    mines = min(INITIAL_MINES + curriculum_level * 2, MAX_MINES)
    
    print(f"Training on grid size: {width}x{height} with {mines} mines")
    
    online_net = DualHeadNetwork(width, height).to(device)
    target_net = DualHeadNetwork(width, height).to(device)
    target_net.load_state_dict(online_net.state_dict())
    
    optimizer = torch.optim.Adam(online_net.parameters(), lr=0.0001)
    replay_memory = ReplayBuffer(100000)
    
    batch_size = 128
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.001
    episodes = 1000
    target_update_freq = 10
    
    stats = []
    
    # Pre-fill replay memory with some random experiences
    game = minesweeper_game(width, height, mines)
    for _ in range(1000):
        state = get_state(game).to(device)
        action = random.randint(0, width*height*2-1)
        new_state, reward, done = game_step(game, action, state)
        replay_memory.add((state, action, reward, done, new_state))
        if done:
            game = minesweeper_game(width, height, mines)
    
    for episode in range(episodes):
        game = minesweeper_game(width, height, mines)
        state = get_state(game).to(device)
        total_reward = 0
        steps = 0
        win = False
        
        while True:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-epsilon_decay * episode)
            
            if random.random() > epsilon:
                with torch.no_grad():
                    action, _ = online_net.act(state)
            else:
                action = random.randint(0, width*height*2-1)
            
            new_state, reward, done = game_step(game, action, state)
            if done and reward == 10.0:
                win = True
            total_reward += reward
            replay_memory.add((state, action, reward, done, new_state))
            state = new_state
            steps += 1
            
            # Training step
            if len(replay_memory.buffer) >= batch_size:
                experiences = replay_memory.sample(batch_size)
                
                states = torch.stack([torch.as_tensor(e[0], dtype=torch.float32) for e in experiences]).to(device)
                actions = torch.tensor([e[1] for e in experiences], dtype=torch.int64).to(device)
                rewards = torch.tensor([e[2] for e in experiences], dtype=torch.float32).to(device)
                dones = torch.tensor([e[3] for e in experiences], dtype=torch.float32).to(device)
                next_states = torch.stack([torch.as_tensor(e[4], dtype=torch.float32) for e in experiences]).to(device)
                
                # Get current Q values and state values
                current_q_values, current_state_values = online_net(states)
                current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
                
                # Get next state values from target network
                with torch.no_grad():
                    next_q_values, next_state_values = target_net(next_states)
                    next_q_values = next_q_values.max(1)[0]
                    target_values = rewards + gamma * (1 - dones) * next_q_values
                    target_values = target_values.unsqueeze(1)
                
                # Compute losses
                q_loss = F.smooth_l1_loss(current_q_values, target_values)
                value_loss = F.mse_loss(current_state_values, target_values)
                total_loss = q_loss + value_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), 1.0)
                optimizer.step()
            
            if done or steps > width * height:
                break
        
        if episode % target_update_freq == 0:
            target_net.load_state_dict(online_net.state_dict())
        
        stats.append((episode, win, steps, total_reward))
        print(f"Episode {episode}: steps={steps}, won={win}, reward={total_reward:.2f}, epsilon={epsilon:.3f}")
        
        # Check if curriculum should advance
        if episode > 0 and episode % 200 == 0:
            # recent_rewards = [s[2] for s in stats[-100:]]
            # avg_reward = sum(recent_rewards) / len(recent_rewards)
            # if avg_reward > 0:  # If average reward is positive, we can try a harder difficulty
            #     return online_net, stats, True
            game_outcomes = [s[1] for s in stats[-100:]]
            win_rate = sum(game_outcomes) / len(game_outcomes)
            if win_rate > 0.2:
                return online_net, stats, True
    
    return online_net, stats, False

# Main training loop with curriculum learning
current_level = 0
all_stats = []

while current_level * 2 + INITIAL_WIDTH <= MAX_WIDTH:
    print(f"\nStarting curriculum level {current_level}")
    model, level_stats, should_advance = train(current_level)
    all_stats.extend(level_stats)
    
    # Save stats
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    torch.save(model.state_dict(), f"C:/Users/omerk/Documents/GitHub/mINE/python/minesweeper/models/checkpoints/level_{current_level}_{datetime_str}.pth")
    
    with open(f"C:/Users/omerk/Documents/GitHub/mINE/python/minesweeper/models/checkpoints/stats_level_{current_level}_{datetime_str}.csv", 'w', newline='') as tf:
        writer = csv.writer(tf)
        writer.writerow(['episode', 'win', 'steps', 'reward'])
        writer.writerows(level_stats)
    
    if not should_advance:
        print(f"Training plateaued at level {current_level}")
        break
    
    current_level += 1