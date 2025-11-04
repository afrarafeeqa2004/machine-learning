import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob

corpus_content = """
The cat sat on the mat.She is reading a book.They quickly ran to the store.John plays football every Sunday.The beautiful garden smells amazing.Although it was raining, we went for a walk.Can you help me with this problem?
"""
with open('file.txt', 'w') as f:
    f.write(corpus_content.strip())

with open('file.txt', 'r') as f:
    text_corpus = f.read()

print("Libraries imported and 'file.txt' loaded to 'text_corpus' variable.")

tokens = word_tokenize(text_corpus)

print("Tokens (First 30)")
print(tokens[:30])
corrected_tokens = []
corrected_text = []

blob = TextBlob(text_corpus)
corrected_text_corpus = str(blob.correct())

corrected_tokens = word_tokenize(corrected_text_corpus)

print("Corrected Tokens (First 10)")
print(corrected_tokens[:10])

print("\nCorrected Text Corpus")
print(corrected_text_corpus)
# Apply POS tags to each corrected token
pos_tags = nltk.tag.pos_tag(corrected_tokens)

print("POS Tags (First 15)")
print(pos_tags[:15])
english_stop_words = set(stopwords.words('english'))

filtered_tokens = [token for token in corrected_tokens if token.lower() not in english_stop_words and token.isalpha()]

print("Filtered Tokens (Stop Words Removed - First 20)")
print(filtered_tokens[:20])
# Apply stemming and lemmatization to the corrected token list

# Initialize the stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

print("Stemmed Tokens (First 20)")
print(stemmed_tokens[:20])

print("\nLemmatized Tokens (First 20)")
print(lemmatized_tokens[:20])
sentences = sent_tokenize(text_corpus)

total_sentences = len(sentences)

print("Sentence Boundary Detection")
print(f"Total number of sentences detected: <<<{total_sentences}>>>")
print("\nAll Sentences:")
for i, sent in enumerate(sentences):
    print(f"{i+1}. {sent}")
