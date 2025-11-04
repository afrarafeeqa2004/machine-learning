import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import re

newsgroups_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
df = pd.DataFrame({'text': newsgroups_data.data, 'target': newsgroups_data.target})
print("Sample data:\n", df.head(), "\n")
# Clean the text data
def clean_text(text):
    text = text.lower()                          
    text = re.sub(r'\n', ' ', text)              
    text = re.sub(r'[^a-z\s]', '', text)         
    text = re.sub(r'\s+', ' ', text).strip()     
    return text

df['clean_text'] = df['text'].apply(clean_text)
print("Cleaned data sample:\n", df['clean_text'].head(), "\n")
bow_vectorizer = CountVectorizer(stop_words='english', max_features=10000)
bow_matrix = bow_vectorizer.fit_transform(df['clean_text'])

bow_sum = np.array(bow_matrix.sum(axis=0)).flatten()
bow_freq = pd.DataFrame({'word': bow_vectorizer.get_feature_names_out(), 'count': bow_sum})
bow_top20 = bow_freq.sort_values(by='count', ascending=False).head(20)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_text'])

tfidf_sum = np.array(tfidf_matrix.sum(axis=0)).flatten()
tfidf_freq = pd.DataFrame({'word': tfidf_vectorizer.get_feature_names_out(), 'tfidf_score': tfidf_sum})
tfidf_top20 = tfidf_freq.sort_values(by='tfidf_score', ascending=False).head(20)
# Compare both models — visualize
plt.figure(figsize=(14, 6))

# Bag of Words
plt.subplot(1, 2, 1)
sns.barplot(x='count', y='word', data=bow_top20, palette='viridis')
plt.title("Top 20 Words - Bag of Words")
plt.xlabel("Frequency")
plt.ylabel("Word")

# TF-IDF
plt.subplot(1, 2, 2)
sns.barplot(x='tfidf_score', y='word', data=tfidf_top20, palette='plasma')
plt.title("Top 20 Words - TF-IDF")
plt.xlabel("TF-IDF Score")
plt.ylabel("Word")

plt.tight_layout()
plt.show()


# Display Top 20 words comparison table
comparison = pd.merge(
    bow_top20, tfidf_top20, on='word', how='outer'
).fillna(0).sort_values(by='count', ascending=False)
print("\nTop 20 word comparison (BoW vs TF-IDF):\n")
print(comparison)
