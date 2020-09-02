import numpy as np
import tensorflow as tf


class Decoder(tf.keras.layers.Layer):
    """
    Custom tf.keras.layers.Layer Decoder object supports fully connected (fc) and convolutional (cnn) decoding layers.
    """
    def __init__(
        self,
        input_shape,
        latent_dim,
        mode='fc'
    ):
        super(Decoder, self).__init__()

        if mode == 'fc':
            self.out = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(100, activation='relu'),
                tf.keras.layers.Dense(200, activation='relu'),
                tf.keras.layers.Dense(np.prod(input_shape), activation='sigmoid'),
                tf.keras.layers.Reshape(input_shape)
            ])
        elif mode == 'cnn':
            self.out = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    activation='relu'
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    activation='relu'
                ),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1,
                    kernel_size=3,
                    strides=1,
                    padding='same'
                ),
            ])

    def call(self, x, **kwargs):
        return self.out(x)
