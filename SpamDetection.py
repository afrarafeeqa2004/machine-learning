#import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

#load dataset
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/spam_ham_dataset.csv")

print(data.head())
print(data.info())

X = data["text"]
y = data["label_num"]

#split the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

count_vect = CountVectorizer(stop_words="english")
X_train_counts = count_vect.fit_transform(X_train)
X_test_counts = count_vect.transform(X_test)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

model = MultinomialNB()

#train the model
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

#calculate metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Ham", "Spam"],
            yticklabels=["Ham", "Spam"])
plt.title("Confusion Matrix - Spam Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

def predict_message(message):
    msg_counts = count_vect.transform([message])
    msg_tfidf = tfidf_transformer.transform(msg_counts)
    prediction = model.predict(msg_tfidf)[0]
    return "Spam" if prediction == 1 else "Ham"
            
print(predict_message("Congratulations! You won a $1000 prize. Click here!"))
print(predict_message("Are we still meeting tomorrow at 10am?"))
