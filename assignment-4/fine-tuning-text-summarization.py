!pip install transformers datasets sentencepiece accelerate -q

from datasets import load_dataset

dataset = load_dataset("cnn_dailymail", "3.0.0")

from transformers import T5TokenizerFast

tokenizer = T5TokenizerFast.from_pretrained("t5-small")

max_in = 512
max_out = 128

def preprocess(batch):
    #encode input
    inputs = ["summarize: " + art for art in batch["article"]]
    model_inputs = tokenizer(
        inputs,
        max_length=max_in,
        truncation=True,
        padding="max_length"
    )

    #encode labels
    labels = tokenizer(
        batch["highlights"],
        max_length=max_out,
        truncation=True,
        padding="max_length"
    )

    labels_ids = labels["input_ids"]
    for i in range(len(labels_ids)):
        labels_ids[i] = [
            -100 if token == tokenizer.pad_token_id else token
            for token in labels_ids[i]
        ]

    model_inputs["labels"] = labels_ids
    return model_inputs

tokenized = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)

from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments

model = T5ForConditionalGeneration.from_pretrained("t5-small")

args = TrainingArguments(
    output_dir="t5_cnn_final",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=3e-4,
    num_train_epochs=1,
    weight_decay=0.01,
    fp16=True,
    eval_strategy="no",
    logging_steps=50,
)

#create trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"].select(range(20000)),
)

#train the model
trainer.train()

#save the model
trainer.save_model("t5_cnn_final")
tokenizer.save_pretrained("t5_cnn_final")

import torch

def summarize(text, max_len=150, min_len=30):
    input_text = "summarize: " + text

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_len,
            min_length=min_len,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

from datasets import load_dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

article = dataset["test"][0]["article"]

print("ARTICLE:\n", article[:400], "...\n")
print("SUMMARY:\n", summarize(article))
