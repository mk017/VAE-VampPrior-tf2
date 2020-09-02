import numpy as np
import tensorflow as tf


def log_normal_pdf(x, mean, logvar, axis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((x - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=axis
    )


def preprocess_images(images, apply_filter=False):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    if apply_filter:
        return np.where(images > .5, 1.0, 0.0).astype('float32')
    else:
        return images.astype('float32')
