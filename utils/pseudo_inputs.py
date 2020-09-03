import numpy as np
import tensorflow as tf


class TrainablePseudoInputs(tf.keras.layers.Layer):
    def __init__(self, batch_size_u, activation="hard_sigmoid", **kwargs):
        self.batch_size_u = batch_size_u
        self.pseudo_inputs = None
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.pseudo_inputs = self.add_weight(
            shape=(self.batch_size_u, np.prod(input_shape[1:])),
            initializer=tf.random_normal_initializer(mean=-0.05, stddev=0.01),
            dtype=tf.float32,
            name='u'
        )
        self.reshape_image = tf.keras.layers.Reshape(input_shape[1:])
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.activation(self.reshape_image(self.pseudo_inputs))

    def compute_output_shape(self, input_shape):
        return self.batch_size_u, input_shape[1:]
