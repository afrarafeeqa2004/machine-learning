import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train_flat = x_train.reshape((len(x_train), -1))
x_test_flat = x_test.reshape((len(x_test), -1))
input_dim = x_train_flat.shape[1]  # 28*28 = 784

def build_autoencoder(input_dim, latent_dim):
    input_img = Input(shape=(input_dim,))
    encoded = Dense(latent_dim, activation='relu')(input_img)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)

    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder
latent_dims = [2, 10, 50]
reconstructions = {}

for latent_dim in latent_dims:
    print(f"\nTraining autoencoder with latent_dim={latent_dim}")
    autoencoder, encoder = build_autoencoder(input_dim, latent_dim)
    autoencoder.fit(x_train_flat, x_train_flat,
                    epochs=20,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test_flat, x_test_flat),
                    verbose=0)
    x_test_recon = autoencoder.predict(x_test_flat)
    reconstructions[latent_dim] = x_test_recon
n = 10
plt.figure(figsize=(12, 6))

for i in range(n):
    ax = plt.subplot(3, n, i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.axis('off')
    if i == 0:
        plt.ylabel("Original")
    ax = plt.subplot(3, n, i+1+n)
    plt.imshow(reconstructions[2][i].reshape(28,28), cmap='gray')
    plt.axis('off')
    if i == 0:
        plt.ylabel("Latent=2")
    ax = plt.subplot(3, n, i+1+2*n)
    plt.imshow(reconstructions[10][i].reshape(28,28), cmap='gray')
    plt.axis('off')
    if i == 0:
        plt.ylabel("Latent=10")

plt.suptitle("Original vs Reconstructed Images")
plt.show()
