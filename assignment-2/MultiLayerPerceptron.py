import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

tf.config.run_functions_eagerly(True)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0


def create_mlp():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

splits = [0.8, 0.7, 0.6]
optimizer_classes = {
    "SGD": SGD,
    "Adam": Adam,
    "RMSprop": RMSprop
}
results = {}
for split in splits:
    print(f"\n--- Train-Test Split: {int(split*100)}% ---")
    X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=(1-split), random_state=42)

    for opt_name, opt_class in optimizer_classes.items():
        print(f"\nTraining with optimizer: {opt_name}")

        #Create a new optimizer instance for each new model
        optimizer = opt_class(learning_rate=0.001 if opt_name != "SGD" else 0.01)

        model = create_mlp()
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(X_train, Y_train, epochs=5, batch_size=128, verbose=0, validation_data=(X_val, Y_val))

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        results[(split, opt_name)] = test_acc
        print(f"Test Accuracy with {opt_name}: {test_acc:.4f}")

plt.figure(figsize=(8, 5))
for split in splits:
    accs = [results[(split, opt)] for opt in optimizer_classes.keys()]
    plt.plot(list(optimizer_classes.keys()), accs, marker='o', label=f'Train {int(split*100)}%')

plt.title("Optimizer vs Test Accuracy for Different Train-Test Splits")
plt.xlabel("Optimizer")
plt.ylabel("Test Accuracy")
plt.legend()
plt.grid(True)
plt.show()
