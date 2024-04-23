import tensorflow as tf

own_cards_input = tf.keras.layers.Input(shape=(23, ), name="Own Cards") # 23 cards
rounds_input = tf.keras.layers.Input(shape=(20, 23+5+5, ), name="Rounds History") # 20 rounds of 23 cards, plus who guessed and who had the cards
flat_rounds = tf.keras.layers.Flatten()(rounds_input)
x = tf.keras.layers.concatenate([own_cards_input, flat_rounds])
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
best_guess = tf.keras.layers.Dense(23, activation='softmax', name="Outputs")(x)

model = tf.keras.Model(inputs=[own_cards_input, rounds_input], outputs=best_guess)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=[
        tf.keras.losses.BinaryCrossentropy(from_logits=True),
    ]
)

# own_cards = tf.convert_to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
own_cards = tf.zeros(shape=(23,))
rounds = tf.zeros(shape=(20, 23+5+5, ))

def predict(own_cards, rounds):
  inputs = tf.concat([own_cards, tf.reshape(rounds, shape=(660, ))])
  print(model(inputs, training=False))

predict(own_cards, rounds)