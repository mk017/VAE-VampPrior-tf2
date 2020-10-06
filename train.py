import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from models.vae import Vae
from utils.utilities import preprocess_images

from IPython import display
import pdb

# Training
epochs = 4
batch_size = 32

# Data and preprocessing
apply_filter = False
input_shape = (28, 28, 1)
train_size = 60000
test_size = 10000

# Model architecture
latent_dim = 64
model = 'hvae'
prior = 'vamp'
layer_type = 'bigcnn'
sample_mode = 'mc'

# Load the dataset
(train_images, _), (test_images, _) = tf.keras.datasets.fashion_mnist.load_data()

train_dataset = (tf.data.Dataset.from_tensor_slices(
    preprocess_images(
        train_images,
        apply_filter=apply_filter
    )
).shuffle(train_size).batch(batch_size))

test_dataset = (tf.data.Dataset.from_tensor_slices(
    preprocess_images(
        test_images,
        apply_filter=apply_filter
    )
).shuffle(test_size).batch(batch_size))

# VAE class
vae = Vae(
    input_shape=input_shape,
    latent_dim=latent_dim,
    layer_type=layer_type,
    hierarchical=(model == 'hvae'),
    vampprior=(prior == 'vamp'),
    expected_value=(sample_mode == 'expectation')
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

# set the dimensionality of the latent space to a plane for visualization later
num_examples_to_generate = 16

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim]
)


# Compile and train VAE
import time
optimizer = tf.keras.optimizers.Adam(1e-4)

for epoch in range(1, epochs + 1):
  start_time = time.time()
  for train_x in train_dataset:
    train_step(vae, train_x, optimizer)
  end_time = time.time()

  val_kl_loss = tf.keras.metrics.Mean()
  val_recon_loss = tf.keras.metrics.Mean()
  for test_x in test_dataset:
    recon_loss, kl_loss = vae.compute_loss(test_x)
    val_kl_loss(kl_loss)
    val_recon_loss(recon_loss)
  elbo = -val_kl_loss.result() - val_recon_loss.result()
  display.clear_output(wait=False)
  print(f'Epoch: {epoch} | Test set ELBO: {elbo:.2f}, KL div.: {val_kl_loss.result():.2f}, '
        f'Recon loss: {val_recon_loss.result():.2f} | time: {end_time - start_time:.2f}')



# Evaluate trained VAE
test_images = preprocess_images(test_images)
decoded_imgs = vae(test_images).numpy()
#pseudo_inputs = vae.pseudo_inputs_layer(test_images).numpy()

embedding = vae.predict_embedding(test_images)

# Plot umap
plt.scatter(embedding[:, 0], embedding[:, 1], c=y_test, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title(f'2D-latent space of {data_set}', fontsize=24)
plt.savefig(f'img/{training_id}_embedding.png')

# Plot evaluation
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(np.squeeze(test_images[i], axis=2))
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
plt.savefig(f'img/{model}_{layer_type}_{prior}_{sample_mode}_epochs_{epochs}.png')
plt.show()
