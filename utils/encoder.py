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
        hidden_dim=256,
        mode='fc',
        conditional_on_other_z=False,
        name='encoder'
    ):
        super(Encoder, self).__init__(name=name)

        if mode == 'fc':
            self.layers = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(200, activation='relu'),
                tf.keras.layers.Dense(100, activation='relu'),
            ])
        elif mode == 'cnn':
            self.layers = tf.keras.Sequential([
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
                tf.keras.layers.Conv2D(
                    filters=16,
                    kernel_size=3,
                    strides=(1, 1),
                    activation='relu'
                ),
                tf.keras.layers.Flatten(),
            ])
        elif mode == 'bigcnn':
            self.layers = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=7,
                    strides=(1, 1),
                    activation='relu',
                    padding='same'
                ),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    activation='relu',
                    padding='same'
                ),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=5,
                    strides=(1, 1),
                    activation='relu',
                    padding='same'
                ),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    activation='relu',
                    padding='same'
                ),
                tf.keras.layers.Conv2D(
                    filters=6,
                    kernel_size=3,
                    strides=(1, 1),
                    activation='relu',
                    padding='same'
                ),
                tf.keras.layers.Flatten(),
            ])

        self.distribution_parameter = tf.keras.layers.Dense(2 * latent_dim, activation='linear')

        if conditional_on_other_z:
            self.conditional_layers = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=latent_dim),
                tf.keras.layers.Dense(hidden_dim, activation='relu'),
            ])
            self.joint_layer = tf.keras.layers.Dense(hidden_dim, activation='relu')

    def call(self, x, z=None, **kwargs):
        if z is None:
            x = self.layers(x)
        else:
            x = tf.concat([self.layers(x), self.conditional_layers(z)], axis=1)
            x = self.joint_layer(x)
        return self.distribution_parameter(x)
