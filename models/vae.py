import numpy as np
import tensorflow as tf

from utils.encoder import Encoder
from utils.decoder import Decoder
from utils.utilities import log_normal_pdf

class Vae(tf.keras.Model):
    def __init__(
            self,
            input_shape,
            latent_dim,
            mode="fc"
    ):
        super(Vae, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(
            input_shape=input_shape,
            latent_dim=latent_dim,
            mode=mode
        )
        self.decoder = Decoder(
            input_shape=input_shape,
            latent_dim=latent_dim,
            mode=mode
        )

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
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)

        recon_loss = -tf.reduce_mean(logpx_z)
        kl_loss = -tf.reduce_mean(logpz - logqz_x)
        return recon_loss, kl_loss
        #return -tf.reduce_mean(logpx_z + logpz - logqz_x) = (recon_loss + kl_loss)

    def call(self, x, **kwargs):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_reconstructed = self.decode(z, apply_sigmoid=True)
        return x_reconstructed
