import pandas as pd
import numpy as np
import pandas as pd
import io
from google.colab import files
import matplotlib.pyplot as plt

def f(w,b,x):
  return 1.0/(1.0+np.exp(-(w*x+b)))
# calculate loss/error
def error(w,b):
  err=0.0
  for x,y in zip(X,Y):
    fx=f(w,b,x)
    err+=0.5*(fx-y)**2
  return err


def grad_b(w,b,x,y):
  fx=f(w,b,x)
  return (fx-y)*fx*(1-fx)

def grad_w(w,b,x,y):
  fx=f(w,b,x)
  return (fx-y)*fx*(1-fx)*x
def do_stochastic_gradient_descent(X,Y,w,b,eta,max_epochs):
  Loss=[]
  for i in range(max_epochs):
    dw,db=0,0
    for x,y in zip(X,Y):
      dw+=grad_w(w,b,x,y)
      db+=grad_b(w,b,x,y)
      w=w-eta*dw
      b=b-eta*db
    print('Epoch ',i)
    print('Loss = ',error(w,b))
    Loss.append(error(w,b))
  return w,b,Loss
uploaded=files.upload()
df=pd.read_csv("/content/drive/MyDrive/Colab Notebooks/data.csv")
print(df)
x=df['X']
y=df['Y']

X=np.array(x)
Y=np.array(y)
plt.scatter(X,Y,marker='*')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Function")
# Perform optimization
initial_w=1
initial_b=1
eta=0.01
max_epochs=100

w,b,l=do_stochastic_gradient_descent(X,Y,initial_w,initial_b,eta,max_epochs)
plt.plot(l)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")

est=[]
for x in X:
  fx=f(w,b,x)
  est.append(fx)
plt.scatter(X,Y,marker='*',label='y')
plt.scatter(X,est,marker='o',label='f(x)')
plt.legend()
