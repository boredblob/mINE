from re import I
import torch
from torch import nn
import numpy as np
from collections import deque
import itertools
import random
from datetime import datetime
import csv
import math 

from minesweeper import minesweeper_game

game_width = 30
game_height = 16
num_mines = 99

class Network(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(game_width*game_height*3, 128),
      nn.ReLU(),
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, game_width*game_height*2)
    )

  def forward(self, t):
    t = self.net(t)
    return t
  
  def act(self, state):
    state_t = torch.as_tensor(state, dtype=torch.float32)
    q_values = self(state_t.unsqueeze(0))

    max_q = torch.argmax(q_values, dim=1)[0]
    return max_q.item()

def get_state(game_instance: minesweeper_game):
  # info going into model: distances from mines, whether cells are cleared, 
  mines_grid = torch.asarray(game_instance.mines_grid, dtype=torch.int64)
  mines_grid[mines_grid == -1] = 0
  
  cleared_grid = torch.asarray(game_instance.cleared_grid, dtype=torch.int64)
  flags_grid = torch.asarray(game_instance.flags_grid, dtype=torch.int64)
  state = torch.cat((mines_grid, cleared_grid, flags_grid), 0)
  return state

def game_step(game_instance: minesweeper_game, action, step):
  # return calculate_state(), reward, game_finished?
  num_clears_before = game_instance.cleared_grid.count(True)

  clamped_action = max(0, min(game_width*game_height*2-1, action))
  clear_or_flag = clamped_action // (game_width*game_height)
  idx = action - (clear_or_flag*game_width*game_height)

  game_status = -1 # lose by default if action is somehow invalid
  if clear_or_flag == 0: 
    game_status = game_instance.clear_cell(*game_instance.to_coords(idx))
  if clear_or_flag == 1: 
    game_status = game_instance.flag_cell(*game_instance.to_coords(idx))

  state = get_state(game_instance)
  num_clears_after = game_instance.cleared_grid.count(True)

  done = False
  reward = min(0.5, math.sqrt((num_clears_after - num_clears_before) / (game_width*game_height)))
  # reward -= ((step*0.1) / 256)
  if game_status != 0:
    done = True
    reward = game_status

  reward = max(-1, min(1, reward)) # normalising reward

  return (state, reward, done)

def decay_epsilon(eps_start, eps_end, eps_decay_rate, current_step):
  return eps_end + (eps_start - eps_end) * np.exp(-1 * eps_decay_rate * current_step)

eps_count = 0
batch_size = 64 # tbd
gamma = 0.99

online_net = Network()
target_net = Network()
target_net.load_state_dict(online_net.state_dict())

epsilon_start = 1
epsilon_end = 0.001
epsilon_decay_rate = 0.003

episode_durations = []
optimiser = torch.optim.Adam(online_net.parameters())

# setup memory with random experiences
# replay_memory = deque(maxlen = 50000)
replay_memory = deque(maxlen=50000)
num_random_game_actions = 1000
game_instance = minesweeper_game(game_width, game_height, num_mines)
current_step = 0
for i in range(num_random_game_actions):
  prev_state = get_state(game_instance)
  action = random.randint(0, game_width*game_height*2-1)
  state, reward, done = game_step(game_instance, action, current_step)
  experience = (prev_state, action, reward, done, state)
  replay_memory.append(experience)
  prev_state = state
  current_step += 1

  if done:
    game_instance = minesweeper_game(game_width, game_height, num_mines)
    current_step = 0

# run games for training
num_training_games = 5000
steps_per_game = []
max_steps_per_game = 500
for t in range(num_training_games):
  game_instance = minesweeper_game(game_width, game_height, num_mines)
  prev_state = get_state(game_instance)
  won_game = False
  game_experiences = []

  for step in itertools.count():
    eps_count += 1
    epsilon = decay_epsilon(epsilon_start, epsilon_end, epsilon_decay_rate, eps_count)
  
    action = random.randint(0, game_width*game_height*2-1)
    if random.random() > epsilon: 
      action = online_net.act(prev_state)

    state, reward, done = game_step(game_instance, action, step)
    if done and (reward == 1): won_game = True
    experience = (prev_state, action, reward, done, state)
    game_experiences.append(experience)
    prev_state = state

    if (step < batch_size):
      experiences = game_experiences + random.sample(replay_memory, batch_size-step)
    else:
      experiences = game_experiences  
    states = np.asarray([e[0] for e in experiences])
    actions = np.asarray([e[1] for e in experiences])
    rewards = np.asarray([e[2] for e in experiences])
    dones = np.asarray([e[3] for e in experiences])
    new_states = np.asarray([e[4] for e in experiences])

    states_t = torch.as_tensor(states, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    new_states_t = torch.as_tensor(new_states, dtype=torch.float32)

    q_values = online_net(states_t)
    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
    
    target_q_output = target_net(new_states_t)
    target_q_values = target_q_output.max(dim=1, keepdim=True)[0]

    optimal_q_values = rewards_t + gamma * (1 - dones_t) * target_q_values

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
  for e in game_experiences: replay_memory.append(e)

datetime_str = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
torch.save(online_net.state_dict(), f"C:/Users/omerk/Documents/GitHub/mINE/python/minesweeper/models/{datetime_str}")

with open(f"C:/Users/omerk/Documents/GitHub/mINE/python/minesweeper/models/{datetime_str}.csv", 'w', newline='') as tf:
  file_writer = csv.writer(tf)
  for i in steps_per_game:
    file_writer.writerow(i)