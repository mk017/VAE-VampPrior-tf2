import numpy as np
import tensorflow as tf


def log_normal_pdf(z, z_mean, z_logvar, mean, logvar, expected_value=False, axis=1):
    log2pi = tf.math.log(2. * np.pi)
    if expected_value:
        if mean is None or logvar is None:
            print("Must provide distribution parameter mean and log_var to compute expected value of log density.")

        return tf.reduce_sum(
            -0.5 * ((tf.square(z_mean) + tf.exp(z_logvar) - 2 * z_mean * mean + tf.square(mean)) / tf.exp(logvar)
                    + logvar + log2pi),
            axis=axis
        )
    else:
        return tf.reduce_sum(
            -.5 * ((z - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=axis
        )


def log_sum_of_exponentials(x, axis):
    x_max = tf.math.reduce_max(x, axis=axis)
    x_max_expanded = tf.expand_dims(x_max, axis=axis)
    return x_max + tf.math.log(tf.reduce_sum(tf.exp(x - x_max_expanded), axis=axis))


def preprocess_images(images, apply_filter=False):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    if apply_filter:
        return np.where(images > .5, 1.0, 0.0).astype('float32')
    else:
        return images.astype('float32')
