import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

#load the dataset
df = pd.read_csv("/content/drive/MyDrive/Train.csv")

print(df.head())
print(df['label'].value_counts())

df['true_label'] = df['label'].map({0: "Negative", 1: "Positive"})

#clean the text
def clean_text(text):
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

def bert_predict(text):
    result = sentiment_pipeline(text[:512])[0]
    label = result['label']

    if label.upper().startswith('POS'):
        return 1
    else:
        return 0

df['bert_pred_binary'] = df['cleaned_text'].apply(bert_predict)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#evaluate the model
def evaluate(true, pred):
    print("Accuracy :", accuracy_score(true, pred))
    print("Precision:", precision_score(true, pred))
    print("Recall   :", recall_score(true, pred))
    print("F1 Score :", f1_score(true, pred))
    print("\n")

evaluate(df['label'], df['bert_pred_binary'])

bert_counts = df['bert_pred_binary'].value_counts().sort_index()

labels = ["Negative", "Positive"]   # 0 - Negative, 1 - Positive
values = [bert_counts.get(0, 0), bert_counts.get(1, 0)]

plt.figure(figsize=(7,5))
plt.bar(labels, values)
plt.title("DistilBERT Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

