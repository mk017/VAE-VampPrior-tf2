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
            hierarchical=False,
            layer_type='fc',
            vampprior=False,
            expected_value=False,
            hidden_dim=300
    ):
        super(Vae, self).__init__()
        self.latent_dim = latent_dim
        self.vampprior = vampprior
        self.expected_value = expected_value
        self.hierarchical = hierarchical
        print(f'vampprior: {vampprior}\n'
              f'expected_value: {expected_value}\n'
              f'hierarchical: {hierarchical}')

        self.encoder = Encoder(
            input_shape=input_shape,
            latent_dim=latent_dim if not hierarchical else int(latent_dim/2),
            mode=layer_type,
            name='encoder_z'
        )
        if hierarchical:
            self.encoder_z0 = Encoder(
                input_shape=input_shape,
                latent_dim=int(latent_dim/2),
                mode=layer_type,
                conditional_on_other_z=True,
                name='encoder_z0'
            )
            self.p_z0 = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=int(latent_dim/2)),
                tf.keras.layers.Dense(hidden_dim, activation='relu'),
                tf.keras.layers.Dense(hidden_dim, activation='relu'),
                tf.keras.layers.Dense(latent_dim, activation='relu'),  # mean and logvar each latent_dim/2
            ])

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
        q_mean, q_logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        z = self.reparameterize(q_mean, q_logvar)

        if self.hierarchical:
            q_mean_z0, q_logvar_z0 = tf.split(self.encoder_z0(x, z), num_or_size_splits=2, axis=1)
            z0 = self.reparameterize(q_mean_z0, q_logvar_z0)

        # Reconstruction
            x_logit = self.decode(tf.concat([z, z0], axis=1))
        else:
            x_logit = self.decode(z)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

        # Prior
        if self.vampprior:
            pseudo_inputs = self.pseudo_inputs_layer(x)
            p_mean, p_logvar = tf.split(self.encoder(pseudo_inputs), num_or_size_splits=2, axis=1)
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

        if self.hierarchical:
            # import pdb
            # pdb.set_trace()
            p_mean_z0, p_logvar_z0 = tf.split(self.p_z0(z), num_or_size_splits=2, axis=1)
            logpz0 = log_normal_pdf(z=z0, z_mean=q_mean_z0, z_logvar=q_logvar_z0, mean=p_mean_z0, logvar=p_logvar_z0, expected_value=self.expected_value)

        # Posterior
        logqz_x = log_normal_pdf(z=z, z_mean=q_mean, z_logvar=q_logvar, mean=q_mean, logvar=q_logvar, expected_value=self.expected_value)
        if self.hierarchical:
            logqz0_x = log_normal_pdf(z=z0, z_mean=q_mean_z0, z_logvar=q_logvar_z0, mean=q_mean_z0, logvar=q_logvar_z0, expected_value=self.expected_value)
        # Compute loss
        recon_loss = -tf.reduce_mean(logpx_z)
        if self.hierarchical:
            kl_loss = -tf.reduce_mean(logpz + logpz0 - logqz_x - logqz0_x)
        else:
            kl_loss = -tf.reduce_mean(logpz - logqz_x)
        return recon_loss, kl_loss
        #return -tf.reduce_mean(logpx_z + logpz - logqz_x) = (recon_loss + kl_loss)

    def call(self, x, **kwargs):
        mean_z, logvar_z = self.encode(x)
        z = self.reparameterize(mean_z, logvar_z)
        if self.hierarchical:
            q_mean_z0, q_logvar_z0 = tf.split(self.encoder_z0(x, z), num_or_size_splits=2, axis=1)
            z0 = self.reparameterize(q_mean_z0, q_logvar_z0)
            x_reconstructed = self.decode(tf.concat([z, z0], axis=1), apply_sigmoid=True)
        else:
            x_reconstructed = self.decode(z, apply_sigmoid=True)
        return x_reconstructed