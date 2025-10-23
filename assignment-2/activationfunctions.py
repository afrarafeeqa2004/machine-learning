import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 200)
def step(x):
    return np.where(x >= 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # stability trick
    return exp_x / exp_x.sum()

y_step = step(x)
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_leaky = leaky_relu(x)
y_softmax = softmax(x) 

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.plot(x, y_step, label="Step", color="black")
plt.title("Step Function")
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(x, y_sigmoid, label="Sigmoid", color="blue")
plt.title("Sigmoid")
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(x, y_tanh, label="Tanh", color="red")
plt.title("Tanh")
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(x, y_relu, label="ReLU", color="green")
plt.title("ReLU")
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(x, y_leaky, label="Leaky ReLU", color="purple")
plt.title("Leaky ReLU")
plt.grid(True)

plt.subplot(2, 3, 6)
plt.plot(x, y_softmax, label="Softmax", color="orange")
plt.title("Softmax (over array)")
plt.grid(True) 

plt.tight_layout()
plt.show()
