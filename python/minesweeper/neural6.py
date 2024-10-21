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
INITIAL_MINES = 1

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
        reward = max(0.9, 0.09 * cells_cleared)

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

def train(online_net, target_net, optimizer, curriculum_level=0):
    # Set grid size based on curriculum level
    width = MAX_WIDTH
    height = MAX_HEIGHT
    mines = min(INITIAL_MINES + curriculum_level, MAX_MINES)
    
    print(f"Training on grid size: {width}x{height} with {mines} mines")

    
    # Learning rate scheduler for stability
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=100
    )
    
    replay_memory = ReplayBuffer(50000)  # Reduced buffer size for more recent experiences
    
    batch_size = 64  # Smaller batch size for better stability
    gamma = 0.95  # Slightly reduced discount factor
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.995  # Slower decay rate
    episodes = 50000
    target_update_freq = 5
    
    # Initialize loss tracking
    running_loss = []
    episode_rewards = []
    wins = []
    
    # Pre-fill replay memory with fewer random experiences
    game = minesweeper_game(width, height, mines)
    for _ in range(500):  # Reduced prefill size
        state = get_state(game)
        action = random.randint(0, width*height*2-1)
        new_state, reward, done = game_step(game, action, state)
        replay_memory.add((state, action, reward, done, new_state))
        if done:
            game = minesweeper_game(width, height, mines)
    
    for episode in range(episodes):
        game = minesweeper_game(width, height, mines)
        state = get_state(game)
        total_reward = 0
        steps = 0
        episode_loss = []
        
        while True:
            epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
            
            if random.random() > epsilon:
                with torch.no_grad():
                    action_values, _ = online_net(state.unsqueeze(0))
                    action = action_values.argmax().item()
            else:
                action = random.randint(0, width*height*2-1)
            
            new_state, reward, done = game_step(game, action, state)
            if done:
                wins.append(reward > 0)
                
            total_reward += reward
            replay_memory.add((state, action, reward, done, new_state))
            state = new_state
            steps += 1
            
            # Training step
            if len(replay_memory.buffer) >= batch_size:
                experiences = replay_memory.sample(batch_size)
                
                states = torch.stack([e[0] for e in experiences]).to(device)
                actions = torch.tensor([e[1] for e in experiences], dtype=torch.int64).to(device)
                rewards = torch.tensor([e[2] for e in experiences], dtype=torch.float32).to(device)
                dones = torch.tensor([e[3] for e in experiences], dtype=torch.float32).to(device)
                next_states = torch.stack([e[4] for e in experiences]).to(device)
                
                # Compute Q values with gradient clipping
                current_q_values, current_state_values = online_net(states)
                current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
                
                # Target Q values with double Q-learning
                with torch.no_grad():
                    # Get actions from online network
                    next_q_values, _ = online_net(next_states)
                    next_actions = next_q_values.argmax(1, keepdim=True)
                    
                    # Get Q values from target network
                    target_next_q_values, target_next_state_values = target_net(next_states)
                    next_q_values = target_next_q_values.gather(1, next_actions)
                    
                    # Compute targets with clipping
                    rewards = torch.clamp(rewards, -10, 10)  # Clip rewards
                    target_values = rewards.unsqueeze(1) + gamma * (1 - dones.unsqueeze(1)) * next_q_values
                    target_values = target_values.detach()
                
                # Compute losses with Huber loss for stability
                q_loss = F.huber_loss(current_q_values, target_values, reduction='mean', delta=1.0)
                value_loss = F.huber_loss(current_state_values, target_values, reduction='mean', delta=1.0)
                
                # Combined loss with weighting
                loss = q_loss + 0.5 * value_loss
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=1.0)
                
                optimizer.step()
                episode_loss.append(loss.item())
            
            if done or steps > width * height * 2:
                break
        
        # Soft update of target network
        if episode % target_update_freq == 0:
            with torch.no_grad():
                for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
                    target_param.data.copy_(0.001 * online_param.data + 0.999 * target_param.data)
        
        # Track metrics
        avg_episode_loss = sum(episode_loss) / len(episode_loss) if episode_loss else 0
        running_loss.append(avg_episode_loss)
        episode_rewards.append(total_reward)
        
        # Update learning rate based on average reward
        if len(episode_rewards) >= 100:
            avg_reward = sum(episode_rewards[-100:]) / 100
            scheduler.step(avg_reward)
        
        # Print progress
        if episode % 10 == 0:
            avg_loss = sum(running_loss[-100:]) / len(running_loss[-100:])
            avg_reward = sum(episode_rewards[-100:]) / len(episode_rewards[-100:])
            recent_wins = sum(wins[-10:])
            print(f"Episode {episode}: Loss={avg_loss:.4f}, Reward={avg_reward:.2f}, Epsilon={epsilon:.3f}, wins={recent_wins}")

        if episode % 500 == 0:
            datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            torch.save(online_net.state_dict(), f"C:/Users/omerk/Documents/GitHub/mINE/python/minesweeper/models/checkpoints/{datetime_str}.pth")
        
        # Early stopping check
        if episode > 200 and sum(wins[-100:]) / 100 > 0.2:  # If win rate is good
            return online_net, (running_loss, episode_rewards, wins), True
            
    return online_net, (running_loss, episode_rewards, wins), False

# Main training loop with curriculum learning
current_level = 0

online_net = DualHeadNetwork(MAX_WIDTH, MAX_HEIGHT).to(device)
target_net = DualHeadNetwork(MAX_WIDTH, MAX_HEIGHT).to(device)
target_net.load_state_dict(online_net.state_dict())

# Use a smaller learning rate and add weight decay
optimizer = torch.optim.AdamW(
    online_net.parameters(),
    lr=1e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999)
    )

while current_level * 2 + INITIAL_MINES <= MAX_MINES:
    print(f"\nStarting curriculum level {current_level}")
    model, level_stats, should_advance = train(online_net, target_net, optimizer, current_level)

    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    torch.save(online_net.state_dict(), f"C:/Users/omerk/Documents/GitHub/mINE/python/minesweeper/models/checkpoints/level_{current_level}_{datetime_str}.pth")
    with open(f"C:/Users/omerk/Documents/GitHub/mINE/python/minesweeper/models/checkpoints/stats_level_{current_level}_{datetime_str}.csv", 'w', newline='') as tf:
        writer = csv.writer(tf)
        writer.writerow(['loss', 'reward', 'win'])
        writer.writerows(zip(*level_stats))
    
    if not should_advance:
        print(f"Training plateaued at level {current_level}")
        break
    
    current_level += 1