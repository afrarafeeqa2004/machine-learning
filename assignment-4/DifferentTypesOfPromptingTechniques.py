!pip install transformers accelerate sentencepiece

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

pipe = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=model_name,
    device_map="auto",
    max_new_tokens=200
)

#zero-shot prompting
prompt = "Explain what a microcontroller is."

output = pipe(prompt)[0]["generated_text"]
print(output)

#one-shot prompting
prompt = """
Q: What is a resistor?
A: A resistor restricts current flow.

Q: What is a microcontroller?
A:"""

output = pipe(prompt)[0]["generated_text"]
print(output)

#few-shot prompting

prompt = """
You are a friendly electronics tutor.

Q: What is a diode?
A: A diode allows current to flow in one direction.

Q: What is an inductor?
A: An inductor stores magnetic energy.

Q: What is a capacitor?
A: A capacitor stores electrical energy.

Q: What is a microcontroller?
A:
"""

output = pipe(prompt)[0]["generated_text"]
print(output)

#chain-of-thought prompting

prompt = """
Explain what a microcontroller is.
Think step by step.
"""

output = pipe(prompt)[0]["generated_text"]
print(output)

#role-based prompting

prompt = """
You are an embedded systems engineer.
Explain a microcontroller in very simple language.
"""

output = pipe(prompt)[0]["generated_text"]
print(output)

#context and instruction prompting

context = """
Microcontrollers contain CPU, memory, and peripherals on one chip.
Used in embedded devices.
"""

prompt = f"""
Use the context to explain what a microcontroller is in 3 simple sentences.

Context:
{context}

Answer:
"""

output = pipe(prompt)[0]["generated_text"]
print(output)
