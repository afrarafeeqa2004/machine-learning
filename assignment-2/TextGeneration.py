import tensorflow as tf
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

url = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
path = tf.keras.utils.get_file("shakespeare.txt", origin=url)

with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Dataset length: {len(text):,} characters")

text = text[:300000]
vocab = sorted(set(text))
vocab_size = len(vocab)
print("Unique characters:", vocab_size)

char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text], dtype=np.int32)
seq_length = 100
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    return chunk[:-1], chunk[1:]

dataset = sequences.map(split_input_target)

BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = (dataset
           .shuffle(BUFFER_SIZE)
           .batch(BATCH_SIZE, drop_remainder=True)
           .prefetch(tf.data.AUTOTUNE))
embedding_dim = 256
rnn_units = 512

def build_model(vocab_size, embedding_dim, rnn_units):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(None,)),  # Modern Input API
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(vocab_size, embedding_dim, rnn_units)
model.summary()

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

checkpoint_dir = './training_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

EPOCHS = 10  # Increase to 30–50 for richer output
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
plt.plot(history.history['loss'])
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()


gen_model = build_model(vocab_size, embedding_dim, rnn_units)
checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.weights.h5")))
if not checkpoint_files:
    raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}. Make sure training ran.")
latest = checkpoint_files[-1]
print(f"Loading weights from: {latest}")
gen_model.load_weights(latest)  #No need to call build()

def generate_text(model, start_string, num_generate=500, temperature=1.0):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = predictions[:, -1, :] / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    return start_string + ''.join(text_generated)
start = "To be, or not to be: "
print("\n--- Generated Text (temperature=0.7) ---")
print(generate_text(gen_model, start, num_generate=600, temperature=0.7))

print("\n--- Generated Text (temperature=1.0) ---")
print(generate_text(gen_model, start, num_generate=600, temperature=1.0))
