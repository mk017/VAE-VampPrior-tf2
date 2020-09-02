import numpy as np
import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    """
    Custom tf.keras.layers.Layer Encoder object supports fully connected (fc) and convolutional (cnn) encoding layers.
    """
    def __init__(
        self,
        input_shape,
        latent_dim,
        mode='fc'
    ):
        super(Encoder, self).__init__()

        if mode == 'fc':
            self.out = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(200, activation='relu'),
                tf.keras.layers.Dense(100, activation='relu'),
                tf.keras.layers.Dense(2 * latent_dim, activation='linear')
            ])
        elif mode == 'cnn':
            self.out = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    activation='relu'
                ),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    activation='relu'
                ),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(2 * latent_dim),
            ])

    def call(self, x, **kwargs):
        return self.out(x)
