#import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

#load titanic dataset from seaborn
import seaborn as sns
titanic = sns.load_dataset('titanic')

print(titanic.head())

#handle missing values
#fill missing 'age' with median
titanic['age'].fillna(titanic['age'].median(), inplace=True)

#fill missing 'embarked' with mode
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)

#drop 'deck' column
titanic.drop(columns=['deck'], inplace=True)

#drop 'alive' column
titanic.drop(columns=['alive'], inplace=True)

#encode categorical variables
#convert 'sex' and 'embarked' to numerical using LabelEncoder
label_encoders = {}
for col in ['sex', 'embarked']:
    le = LabelEncoder()
    titanic[col] = le.fit_transform(titanic[col])
    label_encoders[col] = le

#convert 'who', 'class', 'embark_town' using one-hot encoding
titanic = pd.get_dummies(titanic, columns=['who', 'class', 'embark_town'], drop_first=True)

#feature selection
X = titanic.drop('survived', axis=1)
y = titanic['survived']

#split the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
