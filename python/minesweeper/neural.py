# grid of cleared cells as 0-8 -> 0-1
# uncleared = 0
# flagged are appended as 0, 1

import tensorflow as tf
from tensorflow import keras
import numpy as np
from collections import deque
import itertools
import random

from python.minesweeper.minesweeper import minesweeper_game

# model = keras.Sequential([
#   keras.layers.Input((30, 16, 3)),
#   keras.layers.Flatten(),
#   # keras.layers.Dense(128, activation='relu'),
#   # no hidden layers for testing
#   keras.layers.Dense(30*16*2)
# ])

class Network(keras.Model):
  def __init__(self):
    super().__init__()
    self.input = keras.layers.Input((30, 16, 3))
    self.flatten = keras.layers.Flatten()
    self.output = keras.layers.Dense(30*16*2)

  def call(self, inputs):
    x = self.input(inputs)
    x = self.flatten(inputs)
    x = self.output()
    return x
  
  def act(self, state):
    state_t = tf.convert_to_tensor(state, dtype=tf.float32)
    q_values = self(tf.expand_dims(state, 0))

    max_q = tf.argmax(q_values)[0]
    return int(max_q)

def do_game_step(game_instance):
  # return calculate_state(), num cells cleared, game_finished?
  pass

def take_action(model_output_weights):
  # actions are clear/flag for each cell
  pass

def decay_epsilon(eps_start, eps_end, eps_decay_rate, current_step):
  return eps_end + (eps_start - eps_end) * np.exp(-1 * eps_decay_rate * current_step)

eps_count = 0
batch_size = 32 # tbd
gamma = 0.99

online_net = Network()
target_net = Network()
# tbd: copy state from online into target

epsilon_start = 1
epsilon_end = 0.001
epsilon_decay_rate = 0.003

episode_durations = []
optimiser = keras.optimisers.Adam()

replay_memory = deque(maxlen = 50000)
num_games = 100 # -> 1000
for i in range(num_games):
  # init game and get init_state
  # action = random_action
  # state, reward, done = game_step()
  # experience = (init_state, action, reward, done, state)
  # init_state = state

  # if done, reset game
  pass

for t in range(num_games):
  # init_state = game initial state

  for step in itertools.count():
    eps_count += 1
    epsilon = decay_epsilon(epsilon_start, epsilon_end, epsilon_decay_rate, eps_count)
  
  if random.random() <= epsilon:
    action = 0 # random action in space
  else:
    action = online_net() # pass in state

  # state, reward, done = game_step()
  # experience = (init_state, action, reward, done, state)
  # replay_memory.append(experience)
  # init_state = state

  # experiences = random.sample(replay_memory, batch_size)
  # states = np.asarray([e[0] for e in experiences])
  # actions = np.asarray([e[1] for e in experiences])
  # rewards = np.asarray([e[2] for e in experiences])
  # dones = np.asarray([e[3] for e in experiences])
  # new_states = np.asarray([e[3] for e in experiences])

  # states_t = tf.convert_to_tensor(states)
  # actions_t = tf.expand_dims(tf.convert_to_tensor(states), -1)

  # q_values = online_net(states_t)
  # action_q_values = 


game_instance = minesweeper_game(30, 16, 99)

state, reward, done = do_game_step(game_instance)

grid = np.empty((30,16))
grid.fill(0.5)
grid2 = np.empty((30,16))
grid.fill(0.7)
input_tensor = tf.stack((grid, grid2), -1)
input_tensor = tf.reshape(input_tensor, (1, 30, 16, 2))

model.compile()

print(model(input_tensor, training=False))