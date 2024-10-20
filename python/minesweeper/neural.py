# grid of cleared cells as 0-8 -> 0-1
# uncleared = 0
# flagged are appended as 0, 1

import tensorflow as tf
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

model = tf.keras.Sequential([
  tf.keras.layers.Input((28, 28)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])