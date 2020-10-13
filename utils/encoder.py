import numpy as np
import tensorflow as tf


class DensityLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            latent_dim
    ):
        super(DensityLayer, self).__init__(name='density')

        self.dense_mean = tf.keras.layers.Dense(
            units=latent_dim,
            activation="linear"
        )
        self.dense_logvar = tf.keras.layers.Dense(
            units=latent_dim,
            activation="linear"
        )

    def call(self, x, **kwargs):
        mean = self.dense_mean(x)
        logvar = tf.clip_by_value(
            self.dense_logvar(x),
            clip_value_min=np.log(0.001),  # variance larger 0.001
            clip_value_max=np.log(10.0)  # variance smaller 10.0
        )
        return mean, logvar


class Encoder(tf.keras.layers.Layer):
    """
    Custom tf.keras.layers.Layer Encoder object supports fully connected (fc) and convolutional (cnn) encoding layers.
    """
    def __init__(
        self,
        input_shape,
        latent_dim,
        factor=1,
        activation='relu',
        conditional_on_other_z=False,
        name='encoder'
    ):
        super(Encoder, self).__init__(name=name)
        hidden_dim = 256
        self.layers = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv2D(filters=int(8*factor), kernel_size=7, strides=(1, 1), activation=activation, padding='same'),
            tf.keras.layers.Conv2D(filters=int(16*factor), kernel_size=3, strides=(2, 2), activation=activation, padding='same'),
            tf.keras.layers.Conv2D(filters=int(16*factor), kernel_size=5, strides=(1, 1), activation=activation, padding='same'),
            tf.keras.layers.Conv2D(filters=int(32*factor), kernel_size=3, strides=(2, 2), activation=activation, padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(hidden_dim, activation='relu')
        ])

        if conditional_on_other_z:
            self.conditional_layers = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=latent_dim),
                tf.keras.layers.Dense(hidden_dim, activation=activation),
                tf.keras.layers.Dense(hidden_dim, activation=activation),
            ])
            self.joint_layer = tf.keras.layers.Dense(hidden_dim, activation=activation)

        self.distribution_parameter = DensityLayer(latent_dim)

    def call(self, x, z=None, **kwargs):
        if z is None:
            x = self.layers(x)
        else:
            x = tf.concat([self.layers(x), self.conditional_layers(z)], axis=1)
            x = self.joint_layer(x)

        mean, logvar = self.distribution_parameter(x)
        return mean, logvar
