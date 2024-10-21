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
        
        # Increased network capacity and added residual connections
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(128) for _ in range(3)
        ])
        
        # Action head with broader intermediate layers
        self.action_conv = nn.Conv2d(128, 64, kernel_size=1)
        self.action_fc1 = nn.Linear(64 * width * height, 512)
        self.action_fc2 = nn.Linear(512, 256)
        self.action_fc3 = nn.Linear(256, width * height * 2)
        
        # Value head with deeper evaluation
        self.value_conv = nn.Conv2d(128, 64, kernel_size=1)
        self.value_fc1 = nn.Linear(64 * width * height, 512)
        self.value_fc2 = nn.Linear(512, 256)
        self.value_fc3 = nn.Linear(256, 1)

    def forward(self, x):
        # Enhanced state processing
        x = x.view(-1, 3, self.height, self.width)  # Now expecting 3 channels
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Process through residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Action head with dropout for regularization
        action = F.relu(self.action_conv(x))
        action = action.view(-1, 64 * self.width * self.height)
        action = F.dropout(F.relu(self.action_fc1(action)), p=0.2, training=self.training)
        action = F.dropout(F.relu(self.action_fc2(action)), p=0.2, training=self.training)
        action = self.action_fc3(action)
        
        # Value head with dropout
        value = F.relu(self.value_conv(x))
        value = value.view(-1, 64 * self.width * self.height)
        value = F.dropout(F.relu(self.value_fc1(value)), p=0.2, training=self.training)
        value = F.dropout(F.relu(self.value_fc2(value)), p=0.2, training=self.training)
        value = torch.tanh(self.value_fc3(value))
        
        return action, value

    def act(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        action_values, state_value = self(state_t.unsqueeze(0))
        return action_values[0].argmax().item(), state_value.item()

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
    
    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        return random.sample(list(self.buffer), batch_size)

def get_state(game_instance):
    """Enhanced state representation with additional channel for number hints"""
    mines_grid = torch.asarray(game_instance.mines_grid, dtype=torch.float32) / 8
    cleared_grid = torch.asarray(game_instance.cleared_grid, dtype=torch.float32)
    flags_grid = torch.asarray(game_instance.flags_grid, dtype=torch.float32)
    
    # Create hint channel (numbers showing adjacent mines)
    hint_grid = torch.where(cleared_grid == 1, mines_grid, torch.zeros_like(mines_grid))
    
    # Mask unrevealed cells
    uncleared_mask = (cleared_grid == 0) * -1.0
    masked_grid = torch.where(cleared_grid == 1, mines_grid, uncleared_mask)
    
    # Reshape all grids to 2D
    masked_grid_2d = masked_grid.view(game_instance.grid_height, game_instance.grid_width)
    flags_grid_2d = flags_grid.view(game_instance.grid_height, game_instance.grid_width)
    hint_grid_2d = hint_grid.view(game_instance.grid_height, game_instance.grid_width)
    
    # Stack all channels
    state = torch.stack([masked_grid_2d, flags_grid_2d, hint_grid_2d], dim=0).to(device)
    return state

def to_action(action, width, height):
    clamped_action = max(0, min(width*height*2-1, action))
    clear_or_flag = clamped_action // (width*height)
    idx = action - (clear_or_flag*width*height)
    return (clear_or_flag, idx)

def calculate_reward(prev_state, new_state, game_instance, action_type, won, lost, idx):
    """Enhanced reward calculation with better shaping"""
    reward = 0
    
    # Game outcome rewards
    if won:
        reward = 1
    if lost:
        progress_factor = game_instance.num_cleared_cells / (game_instance.grid_width * game_instance.grid_height)
        reward = -1 * (1 - progress_factor)  # Reduced penalty for losses when more progress was made
    
    # Progress rewards
    prev_masked_grid = prev_state[0]
    new_masked_grid = new_state[0]
    prev_cleared = prev_masked_grid[prev_masked_grid < 0].sum().item()
    new_cleared = new_masked_grid[new_masked_grid < 0].sum().item()
    prev_flags = prev_state[1].sum().item()
    new_flags = new_state[1].sum().item()
    cells_cleared = new_cleared - prev_cleared
    flags_placed = new_flags - prev_flags
    # print(prev_state[0], new_state[0])
    # print(prev_cleared, new_cleared, prev_flags, new_flags)
    
    if cells_cleared > 0:
        reward += 0.09 * cells_cleared

    # Penalise inaction
    if flags_placed == 0 and cells_cleared == 0:
        reward -= 0.2
    
    # Flag management rewards/penalties
    if action_type == 1:  # Flag action
        if game_instance.flags_grid[idx]:
            reward += 0.05
        else:
            reward -= 0.05
        if game_instance.num_flagged_cells > game_instance.num_mines:
            reward -= 0.2
    
    # Small step penalty that decreases as the game progresses
    progress = game_instance.num_cleared_cells / (game_instance.grid_width * game_instance.grid_height)
    step_penalty = 0.02 * (1 - progress)
    reward -= step_penalty
    
    return max(0, min(1, reward))

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
    
    reward = calculate_reward(prev_state, new_state, game_instance, clear_or_flag, won, lost, idx)
    
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

    learning_rate = 3e-4  # Slightly reduced learning rate
    batch_size = 256  # Increased batch size
    gamma = 0.99  # Keep the same
    epsilon_start = 1.0
    epsilon_end = 0.05  # Slightly higher minimum exploration
    epsilon_decay = 0.0005  # Slower decay
    target_update_freq = 5  # More frequent target updates
    episodes = 1000

    # batch_size = 128
    # gamma = 0.99
    # epsilon_start = 1.0
    # epsilon_end = 0.01
    # epsilon_decay = 0.001
    # episodes = 1000
    # target_update_freq = 10
    
    optimizer = torch.optim.Adam(online_net.parameters(), lr=learning_rate)
    replay_memory = ReplayBuffer(100000)
    
    
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
        step_loss = 0
        
        while True:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-epsilon_decay * episode)
            
            if random.random() > epsilon:
                with torch.no_grad():
                    action, _ = online_net.act(state)
            else:
                action = random.randint(0, width*height*2-1)
            
            prev_game = str(game)
            prev_mines_grid = game.mines_grid

            new_state, reward, done = game_step(game, action, state)
            if done and reward == 1.0:
                win = True
            # if episode > 10:
            #     print(action, reward)
            #     print(prev_game)
            #     print("\n")
            #     print(prev_mines_grid)
            #     print("\n")
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
                value_loss = F.mse_loss(current_state_values, next_state_values)
                total_loss = q_loss + value_loss
                step_loss += total_loss
                # if episode > 20:
                #     print("loss", q_loss, value_loss)
                
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), 1.0)
                optimizer.step()
            
            if done or steps > width * height:
                break
        
        if episode % target_update_freq == 0:
            target_net.load_state_dict(online_net.state_dict())
        
        average_loss = step_loss / steps
        stats.append((episode, win, steps, total_reward, average_loss))
        print(f"Episode {episode}: steps={steps}, won={win}, reward={total_reward:.2f}, epsilon={epsilon:.3f}, average_loss={average_loss:.3f}")
        
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
        writer.writerow(['episode', 'win', 'steps', 'reward', 'mean loss'])
        writer.writerows(level_stats)
    
    if not should_advance:
        print(f"Training plateaued at level {current_level}")
        break
    
    current_level += 1