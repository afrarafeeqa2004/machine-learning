#import datasets and other packages
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

#load the dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names=iris.feature_names
target_names=iris.target_names
print(X.shape)
print(y.shape)
#split the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#knn classifier
clf = KNeighborsClassifier(n_neighbors = 5)
clf.fit(X_train, y_train)
X_pred=clf.predict(X_test)
y_pred=clf.predict(X_test)

#calculate metrics
accuracy=clf.score(X_test,y_test)
print(accuracy)
con_matrix=confusion_matrix(y_test,y_pred)
print(con_matrix)
class_report=classification_report(y_test,y_pred)
print(class_report)
