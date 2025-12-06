#retrieval-based chatbot
!pip install pypdf sentence-transformers faiss-cpu transformers

from google.colab import files
uploaded = files.upload()

pdf_path = list(uploaded.keys())[0]
print("Uploaded PDF:", pdf_path)

from pypdf import PdfReader

def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

document_text = extract_pdf_text(pdf_path)
print(document_text[:1000])   # preview

def chunk_text(text, chunk_size=300, overlap=20):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks

chunks = chunk_text(document_text)
len(chunks)

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

def rag_query(query, top_k=3):
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec), top_k)

    retrieved_chunks = "\n".join([chunks[i] for i in indices[0][:2]])  # top 2 instead of 3

    prompt = (
        f"Context:\n{retrieved_chunks}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    output = generator(
    prompt,
    max_length=200,
    num_return_sequences=1,
    no_repeat_ngram_size=3,  # prevents repeating phrases
    do_sample=True,
    temperature=0.7,
    top_k=50
)
    return output[0]['generated_text']

response = rag_query("What is technology")
print(response)

#generative chatbot
!pip install transformers

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

def chatbot_response(user_input):
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    output_ids = model.generate(
        input_ids,
        max_length=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        no_repeat_ngram_size=3
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

#test
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print("Chatbot:", response)

