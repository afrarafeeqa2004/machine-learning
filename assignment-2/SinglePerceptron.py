import numpy as np
import math

# sigmoid (logistic function)
def logistic(x):
  return 1/(1 + np.exp(x))

# softmax function
def softmax(x):
  a = np.exp(x)
  a = a/np.sum(a)
  return a

#the input to the network
x = [1.5, 2.5, 3]

#the weights in the first layer
W1 = np.full((3,4),0.05)

#the weights in the second layer
W2 = np.full((4,3),0.025)

#the weights in the second layer
W3= np.full((3,5),1.0)

#the biases in the 3 layers
b1 = np.asarray([0.1,0.2,0.3,0.4])
b2 = np.asarray([5.2,3.2,4.3])
b3 = np.asarray([0.2,0.45,0.75,0.55,0.95])

# compute the pre activation and activation for the first layer
a1 = np.matmul(x,W1) + b1 
h1 = logistic(a1) 
print('a1')
print(a1)
print('h1')
print(h1)

# compute the pre activation and activation for the second layer
# Note : the input to the second layer is the output from first layer

a2 = np.matmul(h1,W2) + b2 
h2 = logistic(a2)
print('a2')
print(a2)
print('h2')
print(h2)
# compute the pre activation and activation for the second layer
# Note : the input to the second layer is the output from first layer

a3 = np.matmul(h2,W3) + b3 
print('a3')
print(a3)

o = softmax(a3) 
print('Output = ')
print(o)
