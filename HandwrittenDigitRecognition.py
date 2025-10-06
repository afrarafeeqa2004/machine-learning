#import libraries
from tensorflow.keras.datasets import mnist
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import numpy as np

#load the datset and split the dataset for training and testing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalize the values
x_train = x_train.reshape(len(x_train), -1) / 255.0
x_test = x_test.reshape(len(x_test), -1) / 255.0
x_train_small = x_train[:10000]
y_train_small = y_train[:10000]
x_test_small = x_test[:2000]
y_test_small = y_test[:2000]

svm_clf = SVC(kernel='rbf', gamma=0.05, C=5)

#train the model
svm_clf.fit(x_train_small, y_train_small)
y_pred = svm_clf.predict(x_test_small)

#calculate metrics
print("Test Accuracy:", accuracy_score(y_test_small, y_pred))
con_matrix=confusion_matrix(y_test_small,y_pred)
print(con_matrix)
class_report=classification_report(y_test_small,y_pred)
print(class_report)
