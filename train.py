import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

import datetime, time
import argparse
import os
from models.vae import Vae
from utils.utilities import preprocess_images

import pickle


def parse_args():
    desc = "Tensorflow 2.0 implementation of a VAE with a VampPrior"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', type=str, default='vae',
                        help='Type of autoencoder: [vae, hvae]')
    parser.add_argument('--prior', type=str, default='standard',
                        help='Type of prior: [standard, vamp]')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of train set run-throughs.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for SGD.')
    parser.add_argument('--data_set', type=str, default='mnist',
                        help='Data set: [mnist, fashion_mnist]')
    parser.add_argument('--latent_dim', type=int, default=2,
                        help='Dimension of the latent space or bottleneck.')
    parser.add_argument('--layer_type', type=str, default='fc',
                        help='Layer type used in encoder and decoder: [fc, cnn]')
    parser.add_argument('--sample_mode', type=str, default='mc',
                        help='Mode to estimate KL divergence: [mc, expectation]')
    return parser.parse_args()


args = parse_args()
print(args)

# Data and preprocessing
apply_filter = False
input_shape = (28, 28, 1)
train_size = 60000
val_size = 10000
training_id = f'{args.data_set}_{args.model}_{args.layer_type}_{args.prior}_{args.sample_mode}_ldim{args.latent_dim}_epochs{args.epochs}'

# Load the dataset
if args.data_set == 'fashion_mnist':
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
    layer_type=args.layer_type,
    hierarchical=(args.model == 'hvae'),
    vampprior=(args.prior == 'vamp'),
    expected_value=(args.sample_mode == 'expectation'),
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

results = {
    'train_kl_loss': [],
    'train_recon_loss': [],
    'val_kl_loss': [],
    'val_recon_loss': []
}
for epoch in range(1, args.epochs + 1):
    # Train step
    start_time = time.time()
    for step, train_x in tqdm(enumerate(train_dataset), total=int(train_size / args.batch_size)):
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
    results['train_kl_loss'].append(train_kl_loss.result())
    results['train_recon_loss'].append(train_recon_loss.result())
    results['val_kl_loss'].append(val_kl_loss.result())
    results['val_recon_loss'].append(val_recon_loss.result())

    train_kl_loss.reset_states()
    train_recon_loss.reset_states()
    val_kl_loss.reset_states()
    val_recon_loss.reset_states()

# Save results dictionary
with open(f'logs/{training_id}_results.pickle', 'wb') as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

# Evaluate trained VAE
val_images = preprocess_images(val_images)
decoded_imgs = vae(val_images).numpy()
#pseudo_inputs = vae.pseudo_inputs_layer(val_images).numpy()

embedding = vae.predict_embedding(val_images)

# Plot umap
plt.scatter(embedding[:, 0], embedding[:, 1], c=y_val, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title(f'2D-latent space of {args.data_set}', fontsize=24)
plt.savefig(f'img/{training_id}_embedding.png')

# Plot evaluation
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(np.squeeze(val_images[i], axis=2))
    plt.title('original')
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(np.squeeze(decoded_imgs[i], axis=2))
    plt.title('reconstructed')
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig(f'img/{training_id}.png')
plt.show()
