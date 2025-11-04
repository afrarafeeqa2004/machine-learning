import pandas as pd
import numpy as np
import re
from textblob import TextBlob

df = pd.read_csv("Test.csv")

print("Dataset loaded successfully!")
print("Shape of dataset:", df.shape)
print(df.head(), "\n")
print("Columns:", df.columns, "\n")
text_data = df['text']
print("Sample Text:\n", text_data.head(), "\n")
def clean_text(text):
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

cleaned_text = text_data.apply(clean_text)
print("Cleaned Text Sample:\n", cleaned_text.head(), "\n")
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    return polarity

sentiment_scores = cleaned_text.apply(get_sentiment)

sentiment_df = pd.DataFrame({
    'cleaned_text': cleaned_text,
    'sentiment_score': sentiment_scores
})

print("Sentiment DataFrame created!\n")
print(sentiment_df.head(), "\n")
final_df = pd.concat([df, sentiment_df], axis=1)
print("DataFrames joined successfully!\n")
print(final_df.head(), "\n")
def get_sentiment_label(score):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

final_df['sentiment_label'] = final_df['sentiment_score'].apply(get_sentiment_label)

positive_df = final_df[final_df['sentiment_label'] == "Positive"]
negative_df = final_df[final_df['sentiment_label'] == "Negative"]
neutral_df  = final_df[final_df['sentiment_label'] == "Neutral"]

print("Sentiment Distribution:")
print(final_df['sentiment_label'].value_counts(), "\n")

print("Positive Reviews Sample:\n", positive_df.head(), "\n")
print("Negative Reviews Sample:\n", negative_df.head(), "\n")
print("Neutral Reviews Sample:\n", neutral_df.head(), "\n")
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))
sns.countplot(x='sentiment_label', data=final_df, palette='viridis')
plt.title("IMDB Review Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()
