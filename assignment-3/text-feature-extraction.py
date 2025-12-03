#BagOfWords

import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer

#initialize the data
corpus = ['Data Science is an overlap between Arts and Science', 'Generally, Arts graduates are right brained and Science graduates are left-brained','Excelling in both Arts and Science at a time becomes difficult','Natural Language Processing is a part of Data Science']

#use CounterVectorizer function to create BoW model
bag_of_words_model=CountVectorizer()
print(bag_of_words_model.fit_transform(corpus).todense())
bag_of_word_df=pd.DataFrame(bag_of_words_model.fit_transform(corpus).todense())
bag_of_word_df.columns=sorted(bag_of_words_model.vocabulary_)
print(bag_of_word_df.head())

#create a BoW model for the 10 most frequent terms
bag_of_words_model_small=CountVectorizer(max_features=10)
bag_of_words_df_small=pd.DataFrame(bag_of_words_model_small.fit_transform(corpus).todense())
bag_of_words_df_small.columns=sorted(bag_of_words_model_small.vocabulary_)
print(bag_of_words_df_small.head())

#n-gram

#with nltk
import nltk
from nltk import ngrams
list(ngrams('the cute little girl is playing with the kitten.'.split(),2))

#with textblob
import textblob
from textblob import TextBlob

blob=TextBlob("the cute little girl is playing with the kitten.")
import nltk
nltk.download('punkt')

blob.ngrams(n=2)

#TF-IDF

#import libraries
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

#initialize the data
corpus = ['Data Science is an overlap between Arts and Science', 'Generally, Arts graduates are right brained and Science graduates are left-brained','Excelling in both Arts and Science at a time becomes difficult','Natural Language Processing is a part of Data Science']

#use TfidfVectorizer function to create model
tfidf_model=TfidfVectorizer()
print(tfidf_model.fit_transform(corpus).todense())
tfidf_df=pd.DataFrame(tfidf_model.fit_transform(corpus).todense())
print(tfidf_df.columns)
#tfidf_df.columns=sorted(tfidf_model.vocabulary_)
print(tfidf_df.head())

#create a tf-idf model for the 10 most frequent terms
tfidf_model_small=TfidfVectorizer(max_features=10)
tfidf_model_small=pd.DataFrame(tfidf_model_small.fit_transform(corpus).todense())
print(tfidf_model_small.columns)
#tfidf_model_small.columns=sorted(tfidf_model_small.vocabulary_)
print(tfidf_model_small.head())

#NER

!pip install spacy
!python -m spacy download en_core_web_sm

import spacy
from collections import Counter
import pandas as pd

#load NER model
nlp = spacy.load("en_core_web_sm")

#sample text data
texts = [
    "Apple is buying a startup in the UK for $1 billion.",
    "Google plans to invest $10 million in AI research."
]

#extract entities from each sentence
def extract_ner(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

#create NER feature counts (useful for ML models)
def ner_feature_counts(text):
    doc = nlp(text)
    return Counter([ent.label_ for ent in doc.ents])

#process all texts
ner_list = [extract_ner(t) for t in texts]
feature_list = [ner_feature_counts(t) for t in texts]

#convert features into DataFrame
df = pd.DataFrame(feature_list).fillna(0)

print("Named Entities Extracted:")
for i, ents in enumerate(ner_list):
    print(f"\nText {i+1}:")
    for e in ents:
        print(f"  {e[0]} --> {e[1]}")

print("\n\nNER Feature Matrix (for ML models):")
print(df)

#word2vec

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec

nltk.download('punkt')

#read text
with open('/kaggle/input/text-preprocessing-dataset-4/sivakasi.txt', 'r', encoding='utf-8') as f:
    text = f.read().replace("\n", " ")

#tokenize sentences and words
data = [[word.lower() for word in word_tokenize(sent)] for sent in sent_tokenize(text)]

#train CBOW and Skip-gram models
cbow_model = Word2Vec(sentences=data, vector_size=100, window=5, min_count=1, sg=0)
skipgram_model = Word2Vec(sentences=data, vector_size=100, window=5, min_count=1, sg=1)

#get vocabulary words
vocab = list(cbow_model.wv.key_to_index.keys())
print("Sample vocabulary:", vocab[:20])

#calculate similarity
def safe_similarity(model, word1, word2):
    if word1 in model.wv.key_to_index and word2 in model.wv.key_to_index:
        return model.wv.similarity(word1, word2)
    else:
        print(f"One of the words '{word1}' or '{word2}' not in vocabulary.")
        return None

#pick words from the vocabulary for similarity check
word_a = vocab[0]       # first word in vocab
word_b = vocab[1]       # second word in vocab

print("CBOW similarity:", safe_similarity(cbow_model, word_a, word_b))
print("Skip-gram similarity:", safe_similarity(skipgram_model, word_a, word_b))

#glove

!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip

import numpy as np

#load GloVe embeddings
def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

glove_path = "glove.6B.100d.txt"
glove_embeddings = load_glove_embeddings(glove_path)

#test
word = "machine"
print(f"Embedding shape: {glove_embeddings[word].shape}")
print(glove_embeddings[word])

#Fast Text

from gensim.models import FastText
from nltk.tokenize import word_tokenize

sentences = ["FastText embeddings handle subword information.",
             "It is effective for various languages."]
#tokenize sentences
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

#train FastText model
model = FastText(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

#access embeddings
word_embeddings = model.wv
print(word_embeddings['subword'])
