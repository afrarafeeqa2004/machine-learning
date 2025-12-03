#text generation
!pip install transformers torch

from transformers import pipeline

#create a text generation pipeline with DistilGPT-2
generator = pipeline("text-generation", model="distilgpt2")

output = generator(
    "Artificial intelligence will reshape the future because",
    max_length=30,
    do_sample=True,
    num_return_sequences=3,
    temperature=0.7
)
text=output[0]["generated_text"].strip()
print(text)
for i in range(3):
    print(f"\n Output {i+1}")
    text = output[i]["generated_text"].strip()
    print(text)

#chat generation
!pip install transformers torch accelerate

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model = "mistralai/Mistral-7B-Instruct-v0.2"

chatbot = pipeline(
    "text-generation",
    model=model,
    tokenizer=model,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7
)

prompt = "<s>[INST] What is machine learning? Explain simply. [/INST]"

response = chatbot(prompt)

print(f"\n{response[0]['generated_text']}")

#text-summarization
!pip install bert-extractive-summarizer
!pip install spacy
!python -m spacy download en_core_web_sm

#extractive summarizer
text = """
AI, or artificial intelligence, is the ability of computer systems to perform tasks that typically require human intelligence, such as learning, problem-solving, and
decision-making. It uses technologies like machine learning to process vast amounts of data, recognize patterns, and make predictions or decisions without being
explicitly programmed for every scenario.AI systems are designed to simulate human reasoning through logic and math, enabling them to learn from
new information and make decisions based on patterns in existing data.Through machine learning (ML), a subfield of AI, systems learn and improve their performance as
they consume more data.
"""

from summarizer import Summarizer
bert_model = Summarizer()
summary = bert_model(text, min_length=60)
print("Extractive summary")
print(summary)

#abstractive summarizer
from transformers import pipeline

abstractive_summarizer=pipeline("summarization",model="t5-small")
abstractive_summary=abstractive_summarizer(text,max_length=20,min_length=10,do_sample=False)
print("Abstractive summary")
print(abstractive_summary)
