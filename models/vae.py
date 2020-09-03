import numpy as np
import tensorflow as tf

from utils.encoder import Encoder
from utils.decoder import Decoder
from utils.utilities import log_normal_pdf, log_sum_of_exponentials
from utils.pseudo_inputs import TrainablePseudoInputs
class Vae(tf.keras.Model):
    def __init__(
            self,
            input_shape,
            latent_dim,
            layer_type='fc',
            vampprior=False,
            expected_value=False
    ):
        super(Vae, self).__init__()
        self.latent_dim = latent_dim
        self.vampprior = vampprior
        self.expected_value = expected_value

        self.encoder = Encoder(
            input_shape=input_shape,
            latent_dim=latent_dim,
            mode=layer_type
        )
        self.decoder = Decoder(
            input_shape=input_shape,
            latent_dim=latent_dim,
            mode=layer_type
        )
        if self.vampprior:
            self.batch_size_u = 500
            self.pseudo_inputs_layer = TrainablePseudoInputs(self.batch_size_u)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def compute_loss(self, x):
        # Encoding
        q_mean, q_logvar = self.encode(x)
        z = self.reparameterize(q_mean, q_logvar)

        # Reconstruction
        x_logit = self.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

        # Prior
        if self.vampprior:
            pseudo_inputs = self.pseudo_inputs_layer(x)
            p_mean, p_logvar = self.encode(pseudo_inputs)
            logpz = log_normal_pdf(
                z=tf.expand_dims(z, axis=1),
                z_mean=tf.expand_dims(q_mean, axis=1),
                z_logvar=tf.expand_dims(q_logvar, axis=1),
                mean=tf.expand_dims(p_mean, axis=0),
                logvar=tf.expand_dims(p_logvar, axis=0),
                expected_value=self.expected_value,
                axis=2
            )
            # marginalize over batch_size_u
            logpz = log_sum_of_exponentials(logpz, axis=1) - np.log(self.batch_size_u)
        else:
            logpz = log_normal_pdf(z=z, z_mean=q_mean, z_logvar=q_logvar, mean=0.0, logvar=0.0, expected_value=self.expected_value)

        # Posterior
        logqz_x = log_normal_pdf(z=z, z_mean=q_mean, z_logvar=q_logvar, mean=q_mean, logvar=q_logvar, expected_value=self.expected_value)

        # Compute loss
        recon_loss = -tf.reduce_mean(logpx_z)
        kl_loss = -tf.reduce_mean(logpz - logqz_x)
        return recon_loss, kl_loss
        #return -tf.reduce_mean(logpx_z + logpz - logqz_x) = (recon_loss + kl_loss)

    def call(self, x, **kwargs):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_reconstructed = self.decode(z, apply_sigmoid=True)
        return x_reconstructed
