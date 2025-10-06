#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#load dataset
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Titanic-Dataset.csv")
print(data.head())
print(data.info())

data = data.drop(columns=["Name", "Ticket", "Cabin"])

# Fill missing values
data["Age"].fillna(data["Age"].median(), inplace=True)
data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

le = LabelEncoder()
data["Sex"] = le.fit_transform(data["Sex"])
data["Embarked"] = le.fit_transform(data["Embarked"])

X = data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = data["Survived"]

#split the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#standardize the values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=500)

#train the model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#calculate metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

#Survival Count
sns.countplot(ax=axes[0,0], x="Survived", data=data, palette="Set2")
axes[0,0].set_title("Survival Count")

#Survival by Gender
sns.countplot(ax=axes[0,1], x="Sex", hue="Survived", data=data, palette="Set1")
axes[0,1].set_title("Survival by Gender")
axes[0,1].set_xticklabels(["Female (0)", "Male (1)"])

#Survival by Passenger Class
sns.countplot(ax=axes[0,2], x="Pclass", hue="Survived", data=data, palette="coolwarm")
axes[0,2].set_title("Survival by Passenger Class")

#Age Distribution by Survival
sns.histplot(ax=axes[1,0], data=data, x="Age", hue="Survived", multiple="stack", bins=30)
axes[1,0].set_title("Age Distribution by Survival")

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1,1],
            xticklabels=["Did Not Survive", "Survived"],
            yticklabels=["Did Not Survive", "Survived"])
axes[1,1].set_title("Confusion Matrix")

#Feature Importance
features = X.columns
importances = model.coef_[0]
sns.barplot(ax=axes[1,2], x=importances, y=features, palette="viridis")
axes[1,2].set_title("Feature Importance (Logistic Regression)")

plt.tight_layout()
plt.show()
