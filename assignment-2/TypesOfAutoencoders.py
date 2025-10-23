import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model

(x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train_flat = x_train.reshape((len(x_train), -1))
x_test_flat = x_test.reshape((len(x_test), -1))
input_dim = x_train_flat.shape[1]
latent_dim = 64
input_img = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(latent_dim, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)
basic_autoencoder = Model(input_img, decoded)
basic_autoencoder.compile(optimizer='adam', loss='mse')
basic_autoencoder.fit(x_train_flat, x_train_flat,epochs=20,batch_size=256,
                      validation_data=(x_test_flat, x_test_flat))
reconstructed_basic = basic_autoencoder.predict(x_test_flat)
noise_factor = 0.5
x_train_noisy = np.clip(x_train_flat + noise_factor * np.random.normal(size=x_train_flat.shape), 0., 1.)
x_test_noisy = np.clip(x_test_flat + noise_factor * np.random.normal(size=x_test_flat.shape), 0., 1.)

input_noisy = Input(shape=(input_dim,))
encoded_noisy = Dense(128, activation='relu')(input_noisy)
encoded_noisy = Dense(latent_dim, activation='relu')(encoded_noisy)
decoded_noisy = Dense(128, activation='relu')(encoded_noisy)
decoded_noisy = Dense(input_dim, activation='sigmoid')(decoded_noisy)

denoising_autoencoder = Model(input_noisy, decoded_noisy)
denoising_autoencoder.compile(optimizer='adam', loss='mse')
denoising_autoencoder.fit(x_train_noisy, x_train_flat,epochs=20,batch_size=256,
                          validation_data=(x_test_noisy, x_test_flat))
reconstructed_denoise = denoising_autoencoder.predict(x_test_noisy)

intermediate_dim = 256
latent_dim = 64

def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon
class VAE(tf.keras.Model):
    def __init__(self, input_dim, intermediate_dim, latent_dim, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder_h = Dense(intermediate_dim, activation='relu')
        self.z_mean = Dense(latent_dim)
        self.z_log_var = Dense(latent_dim)
        self.decoder_h = Dense(intermediate_dim, activation='relu')
        self.decoder_mean = Dense(input_dim, activation='sigmoid')

    

def call(self, inputs):
        h = self.encoder_h(inputs)
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        z = sampling([z_mean, z_log_var])
        h_decoded = self.decoder_h(z)
        x_decoded_mean = self.decoder_mean(h_decoded)
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        self.add_loss(kl_loss)
  return x_decoded_mean

vae = VAE(input_dim=input_dim, intermediate_dim=intermediate_dim, latent_dim=latent_dim)
vae.compile(optimizer='adam', loss='mse')
vae.fit(x_train_flat, x_train_flat,epochs=20,batch_size=256,validation_data=(x_test_flat, x_test_flat))
reconstructed_vae = vae.predict(x_test_flat)
def plot_reconstruction(original, reconstructed, title, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

plot_reconstruction(x_test_flat, reconstructed_basic, "Basic Autoencoder")
plot_reconstruction(x_test_flat, reconstructed_denoise, "Denoising Autoencoder")
plot_reconstruction(x_test_flat, reconstructed_vae, "Variational Autoencoder")

num_features = 20
num_samples = 1000
X_industrial = np.random.rand(num_samples, num_features)
input_ind = Input(shape=(num_features,))
encoded_ind = Dense(16, activation='relu')(input_ind)
encoded_ind = Dense(8, activation='relu')(encoded_ind)
decoded_ind = Dense(16, activation='relu')(encoded_ind)
decoded_ind = Dense(num_features, activation='sigmoid')(decoded_ind)
industrial_autoencoder = Model(input_ind, decoded_ind)
industrial_autoencoder.compile(optimizer='adam', loss='mse')
industrial_autoencoder.fit(X_industrial, X_industrial, epochs=20, batch_size=32)
