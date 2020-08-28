import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

epochs = 1
# Load the dataset
(x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()

x_train = np.expand_dims(x_train, axis=3).astype('float32') / 255.
x_test = np.expand_dims(x_test, axis=3).astype('float32') / 255.

print(x_train.shape)
print(x_test.shape)


class Encoder(tf.keras.layers.Layer):
    """
    Custom tf.keras.layers.Layer Encoder object supports fully connected (fc) and convolutional (cnn) encoding layers.
    """
    def __init__(
        self,
        input_shape,
        latent_dim,
        mode='fc'
    ):
        super(Encoder, self).__init__()

        if mode == 'fc':
            self.out = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(100, activation='relu'),
                tf.keras.layers.Dense(2 * latent_dim, activation='linear')
            ])
        elif mode == 'cnn':
            self.out = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    activation='relu'
                ),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    activation='relu'
                ),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(2 * latent_dim),
            ])

    def call(self, x, **kwargs):
        return self.out(x)


class Decoder(tf.keras.layers.Layer):
    """
    Custom tf.keras.layers.Layer Decoder object supports fully connected (fc) and convolutional (cnn) decoding layers.
    """
    def __init__(
        self,
        input_shape,
        latent_dim,
        mode='fc'
    ):
        super(Decoder, self).__init__()

        if mode == 'fc':
            self.out = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(2 * latent_dim,)),
                tf.keras.layers.Dense(np.prod(input_shape), activation='sigmoid'),
                tf.keras.layers.Reshape(input_shape)
            ])
        elif mode == 'cnn':
            self.out = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(2 * latent_dim,)),
                tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    activation='relu'
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    activation='relu'
                ),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1,
                    kernel_size=3,
                    strides=1,
                    padding='same'
                ),
            ])

    def call(self, x, **kwargs):
        return self.out(x)


# VAE class
latent_dim = 64
input_shape = (28, 28, 1)
mode = "cnn"
class Ae(tf.keras.models.Model):
    def __init__(self, encoding_dim):
        super(Ae, self).__init__()
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

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


ae = Ae(latent_dim)

# Compile and train VAE
ae.compile(
    optimizer='adam',
    loss=tf.keras.losses.MeanSquaredError()
)

ae.fit(
    x_train, x_train,
    epochs=epochs,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# Evaluate trained VAE
import pdb
#pdb.set_trace()
encoded_imgs = ae.encoder(x_test).numpy()
decoded_imgs = ae.decoder(encoded_imgs).numpy()

# Plot evaluation
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(np.squeeze(x_test[i], axis=2))
    plt.title("original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(np.squeeze(decoded_imgs[i], axis=2))
    plt.title("reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()