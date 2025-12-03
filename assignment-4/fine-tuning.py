!pip install transformers datasets accelerate rouge-score evaluate

from datasets import load_dataset

#load dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

def format_example(example):
    article = example["article"]
    summary = example["highlights"]

    prompt = f"Summarize the following text:\n{article}\n\nSummary:"
    return {"prompt": prompt, "summary": summary}

dataset = dataset.map(format_example)

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"

#load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

#load the model
model = AutoModelForCausalLM.from_pretrained(model_name)

def tokenize(batch):
    texts = [
        p + " " + s
        for p, s in zip(batch["prompt"], batch["summary"])
    ]
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized = dataset.map(
    tokenize,
    batched=True,
    remove_columns=dataset["train"].column_names
)
tokenized.set_format("torch")

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./gpt2-dailymail-summarizer",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    eval_strategy="steps",
    logging_steps=100,
    save_steps=500,
    learning_rate=5e-5,
    fp16=True,
)

#create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
)

#train the model
trainer.train()

trainer.save_model("./gpt2-dailymail-summarizer")
tokenizer.save_pretrained("./gpt2-dailymail-summarizer")

from transformers import pipeline

summarizer = pipeline(
    "text-generation",
    model="./gpt2-dailymail-summarizer",
    tokenizer="./gpt2-dailymail-summarizer",
)

article = """
YOUR LONG CNN/DAILYMAIL ARTICLE TEXT HERE
"""

prompt = (
    "Below is a news article. Provide a short and concise summary.\n\n"
    f"Article:\n{article}\n\nSummary:"
)

result = summarizer(
    prompt,
    max_new_tokens=120,
    no_repeat_ngram_size=3,
    repetition_penalty=1.8,
    do_sample=True,
    top_p=0.92,
    temperature=0.7
)

print(result[0]["generated_text"])
