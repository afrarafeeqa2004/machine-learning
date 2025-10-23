import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

num_words = 10000  # top 10,000 most frequent words
maxlen = 200       # limit each review to 200 words

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words)

#pad sequences to make them the same length
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

model = models.Sequential([
    layers.Embedding(input_dim=num_words, output_dim=128, input_length=maxlen),
    layers.LSTM(128, dropout=0.4, recurrent_dropout=0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#train the model
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_data=(x_test, y_test),
    verbose=1
)
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest Accuracy: {acc:.4f}")

#plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("LSTM Sentiment Analysis Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

#save the trained model
model.save("sentiment_lstm_model.keras")
print("Model saved as sentiment_lstm_model.keras")

#prepare the word index mapping
word_index = tf.keras.datasets.imdb.get_word_index()
reverse_word_index = {value + 3: key for key, value in word_index.items()}
reverse_word_index[0] = "<PAD>"
reverse_word_index[1] = "<START>"
reverse_word_index[2] = "<UNK>"
reverse_word_index[3] = "<UNUSED>"

def encode_review(text):
    words = text.lower().split()
    encoded = [1]  # start token
    for w in words:
        index = word_index.get(w)
        if index is not None and index + 3 < num_words:
            encoded.append(index + 3)
        else:
            encoded.append(2)  # unknown token
    return tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=maxlen)

def predict_sentiment(review):
    sequence = encode_review(review)
    prediction = model.predict(sequence)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    print(f"\nReview: {review}")
    print(f"Predicted Sentiment: {sentiment} (Confidence: {prediction:.4f})")

#test on custom reviews
test_reviews = [
    "The movie was absolutely wonderful, great acting and direction!",
    "I hated the movie. It was boring and too long.",
    "The plot was good but the characters were poorly developed.",
    "An excellent film with a powerful message.",
    "This was the worst movie I have ever seen in my life."
]

for review in test_reviews:
    predict_sentiment(review)
