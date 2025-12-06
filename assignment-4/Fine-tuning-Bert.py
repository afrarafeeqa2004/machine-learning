!pip install transformers datasets accelerate --quiet

from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import torch

#load dataset
dataset = load_dataset("imdb")

#load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch")

train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]

#load the model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

training_args = TrainingArguments(
    output_dir="./bert-imdb",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    optim="adamw_torch",        
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
)

#create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

#train the model
trainer.train()

results = trainer.evaluate()
print(results)

import torch_xla.core.xla_model as xm

if xm.is_master_ordinal():
    xm.save(model.state_dict(), "model_tpu.pt")

import torch_xla.core.xla_model as xm

save_path = "/content/drive/MyDrive/bert_tpu_model.pt"

if xm.is_master_ordinal():
    xm.save(model.state_dict(), save_path)

print("Model saved to:", save_path)

import torch_xla.core.xla_model as xm
device = xm.xla_device()

#model prediction
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    for k in inputs:
        inputs[k] = inputs[k].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    pred = outputs.logits.argmax().item()
    return "Positive" if pred == 1 else "Negative"

print(predict_sentiment("The movie was fantastic!"))
print(predict_sentiment("It was terrible and boring."))
