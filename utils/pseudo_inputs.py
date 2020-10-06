import numpy as np
import tensorflow as tf
from utils.utilities import preprocess_images
import pdb


class SampledPseudoInputsInitializer(tf.keras.initializers.Initializer):
    def __init__(self, n_samples, data_set):
        self.n_samples = n_samples
        # Load the dataset
        if data_set == 'fashion_mnist':
            (train_images, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
        elif data_set == 'mnist':
            (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

        train_images = preprocess_images(train_images, apply_filter=False)
        total_samples = train_images.shape[0]
        self.init_pseudo_inputs = train_images[np.random.choice(total_samples, n_samples), :, :, :]

    def __call__(self, shape, dtype=None):
        return self.init_pseudo_inputs


class TrainablePseudoInputs(tf.keras.layers.Layer):
    def __init__(self, batch_size_u, activation="hard_sigmoid", **kwargs):
        self.batch_size_u = batch_size_u
        self.pseudo_inputs = None
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.pseudo_inputs = self.add_weight(
            shape=(tuple([self.batch_size_u] + input_shape[1:].as_list())),
            initializer=SampledPseudoInputsInitializer(self.batch_size_u),
            #initializer=tf.random_normal_initializer(mean=-0.05, stddev=0.01),
            dtype=tf.float32,
            name='u'
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.activation(self.pseudo_inputs)

    def compute_output_shape(self, input_shape):
        return self.batch_size_u, input_shape[1:]
