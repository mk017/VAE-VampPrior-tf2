import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import datetime, time
import argparse
import os
from models.vae import Vae
from utils.utilities import preprocess_images
from utils.sys_utils import make_directories, pickle_save
from utils.plot_utils import plot_embedding, plot_reconstructed
import umap


def parse_args():
    desc = "Tensorflow 2.0 implementation of a VAE with a VampPrior"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', type=str, default='vae',
                        help='Type of variational autoencoder: [vae, hvae]')
    parser.add_argument('--prior', type=str, default='standard',
                        help='Type of prior: [standard, vamp]')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=250,
                        help='Number of train set run-throughs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for SGD.')
    parser.add_argument('--data_set', type=str, default='mnist',
                        help='Data set: [mnist, fashionmnist]')
    parser.add_argument('--latent_dim', type=int, default=2,
                        help='Dimension of the latent space or bottleneck.')
    parser.add_argument('--network_size', type=float, default=4,
                        help='Factor that determines the number of filters of the convolutional layers.')
    # parser.add_argument('--sample_mode', type=str, default='mc',
    #                     help='Mode to estimate KL divergence: [mc, expectation]')
    parser.add_argument('--gs_key', type=str, default=None,
                        help='Grid search key for saving of results and images.')
    parser.add_argument('--activation', type=str, default='relu',
                        help='Activation function.')
    return parser.parse_args()


args = parse_args()
print(args)

# Data and preprocessing
apply_filter = False
input_shape = (28, 28, 1)
train_size = 60000
val_size = 10000
training_id = f'{args.data_set}_{args.model}_{args.prior}_bs{args.batch_size}_dim{args.latent_dim}_size{args.network_size}_epochs{args.epochs}'

# Load the dataset
if args.data_set == 'fashionmnist':
    (train_images, _), (val_images, y_val) = tf.keras.datasets.fashion_mnist.load_data()
elif args.data_set == 'mnist':
    (train_images, _), (val_images, y_val) = tf.keras.datasets.mnist.load_data()

train_dataset = (tf.data.Dataset.from_tensor_slices(
    preprocess_images(
        train_images,
        apply_filter=apply_filter
    )
).shuffle(train_size).batch(args.batch_size))

val_dataset = (tf.data.Dataset.from_tensor_slices(
    preprocess_images(
        val_images,
        apply_filter=apply_filter
    )
).shuffle(val_size).batch(args.batch_size))


# VAE class
vae = Vae(
    input_shape=input_shape,
    latent_dim=args.latent_dim,
    activation=args.activation,
    filter_factor=args.network_size,
    hierarchical=(args.model == 'hvae'),
    vampprior=(args.prior == 'vamp'),
    data_set=args.data_set
)


@tf.function
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        recon_loss, kl_loss = model.compute_loss(x)
        loss = recon_loss + kl_loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_kl_loss(kl_loss)
    train_recon_loss(recon_loss)


@tf.function
def val_step(model, x):
    """Executes one val step and returns the validation loss.
    """
    recon_loss, kl_loss = model.compute_loss(x)
    val_kl_loss(kl_loss)
    val_recon_loss(recon_loss)


# Compile and train VAE
optimizer = tf.keras.optimizers.Adam(args.lr)

train_kl_loss = tf.keras.metrics.Mean()
train_recon_loss = tf.keras.metrics.Mean()
val_kl_loss = tf.keras.metrics.Mean()
val_recon_loss = tf.keras.metrics.Mean()

history = {
    'train_kl_loss': [],
    'train_recon_loss': [],
    'val_kl_loss': [],
    'val_recon_loss': []
}
for epoch in range(1, args.epochs + 1):
    # Train step
    start_time = time.time()
    for step, train_x in enumerate(train_dataset):
        train_step(vae, train_x, optimizer)
    end_time = time.time()

    # Validation step
    for val_x in val_dataset:
        val_step(vae, val_x)

    # Print validation results
    val_elbo = -val_kl_loss.result() - val_recon_loss.result()
    print(f'Epoch: {epoch} | Val set ELBO: {val_elbo:.2f}, KL div.: {val_kl_loss.result():.2f}, '
        f'Recon loss: {val_recon_loss.result():.2f} | time: {end_time - start_time:.2f}')

    # Log results and reset metrics
    history['train_kl_loss'].append(train_kl_loss.result().numpy())
    history['train_recon_loss'].append(train_recon_loss.result().numpy())
    history['val_kl_loss'].append(val_kl_loss.result().numpy())
    history['val_recon_loss'].append(val_recon_loss.result().numpy())

    train_kl_loss.reset_states()
    train_recon_loss.reset_states()
    val_kl_loss.reset_states()
    val_recon_loss.reset_states()

# compute embedding
val_images = preprocess_images(val_images)
mean, logvar = vae.encode(val_images)
z = vae.reparameterize(mean, logvar)
embedding = {'z': z.numpy(), 'mean': mean.numpy(), 'logvar': logvar.numpy()}

# create directories to save results: history, args and embedding
if args.gs_key is None:
    # create grid search key by date
    args.gs_key = datetime.date.today().strftime("%m%d")
gs_path = os.path.join('gridsearch', args.gs_key)
make_directories(gs_path, ['img', 'results'])

pickle_save(obj=history, path=os.path.join(gs_path, 'results'), filename=f'{training_id}_history')
pickle_save(obj=vars(args), path=os.path.join(gs_path, 'results'), filename=f'{training_id}_args')
pickle_save(obj=embedding, path=os.path.join(gs_path, 'results'), filename=f'{training_id}_embedding')

# pseudo_inputs = vae.pseudo_inputs_layer(val_images).numpy()

# Apply uniform manifold approximation and projection (UMAP) to visualize higher dimensional latent spaces
if args.latent_dim > 2:
    z = umap.UMAP(n_neighbors=30).fit_transform(z)

# Plot embedding z
plot_embedding(
    z, y_val, title_str=f'2D-latent space of {args.data_set}',
    save_str=os.path.join(gs_path, 'img', f'{training_id}_embedding_z'))
# Plot mean embedding
plot_embedding(
    mean, y_val, title_str=f'2D-latent space of {args.data_set}',
    save_str=os.path.join(gs_path, 'img', f'{training_id}_embedding_mean'))

# Plot reconstructed data
decoded_imgs = vae(val_images).numpy()
plot_reconstructed(decoded_imgs, y_val, save_str=os.path.join(gs_path, 'img', f'{training_id}_reconmanifold'))
