#extractive summarization with TextRank
!pip install sumy

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

text = """
Artificial intelligence (AI) is transforming industries across the globe.
From healthcare to finance and transportation, AI-powered technologies are enabling
automation, improving decision-making, and creating new possibilities.
Companies are investing heavily in AI research and development to stay competitive
in the rapidly evolving digital landscape. As AI continues to advance, it is expected
to revolutionize the way people work, live, and interact with technology.
"""

parser = PlaintextParser.from_string(text, Tokenizer("english"))
summarizer = TextRankSummarizer()

summary = summarizer(parser.document, 2)
for sentence in summary:
    print(sentence)

!pip install bert-extractive-summarizer
!pip install spacy
!python -m spacy download en_core_web_sm

#extractive summarizer with hugging face
from summarizer import Summarizer
bert_model = Summarizer()
summary = bert_model(text, min_length=60)
print("Extractive summary")
print(summary)

#abstractive summarizer with hugging face
from transformers import pipeline

text = """
summarize: Artificial intelligence (AI) is transforming industries across the globe.
From healthcare to finance and transportation, AI-powered technologies are enabling
automation, improving decision-making, and creating new possibilities.
Companies are investing heavily in AI research and development to stay competitive
in the rapidly evolving digital landscape. As AI continues to advance, it is expected
to revolutionize the way people work, live, and interact with technology.
"""

abstractive_summarizer=pipeline("summarization",model="t5-small")
abstractive_summary=abstractive_summarizer(text,max_length=20,min_length=10,do_sample=False)
print("Abstractive summary")
print(abstractive_summary)
