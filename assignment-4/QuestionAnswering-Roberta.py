!pip install transformers datasets accelerate --quiet

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset

model_name = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

#load dataset
dataset = load_dataset("squad")

#tokenization and start/end positions
def prepare_features(examples):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,       
        max_length=256,
        stride=64,
        padding="max_length",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    start_positions, end_positions = [], []

    for i, offsets in enumerate(tokenized["offset_mapping"]):
        sample_index = tokenized["overflow_to_sample_mapping"][i]
        answer = examples["answers"][sample_index]

        start_char = answer["answer_start"][0] if answer["answer_start"] else 0
        end_char = start_char + len(answer["text"][0]) if answer["text"] else 0

        sequence_ids = tokenized.sequence_ids(i)
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        # start token
        token_start = next((idx for idx in range(context_start, context_end+1)
                            if offsets[idx][0] <= start_char < offsets[idx][1]), context_start)
        # end token
        token_end = next((idx for idx in range(context_start, context_end+1)
                          if offsets[idx][0] < end_char <= offsets[idx][1]), context_end)

        start_positions.append(token_start)
        end_positions.append(token_end)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    tokenized.pop("offset_mapping")
    return tokenized

#apply preprocessing
tokenized_dataset = dataset.map(
    prepare_features,
    batched=True,
    remove_columns=dataset["train"].column_names
)
tokenized_dataset.set_format("torch")

training_args = TrainingArguments(
    output_dir="./squad_roberta",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    learning_rate=3e-5,
    weight_decay=0.01,
    fp16=True,                  
    dataloader_num_workers=4,    
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="no"     
)

#create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

#train the model
trainer.train()

#save model
trainer.save_model("/content/drive/MyDrive/roberta_squad_model")
print("Model saved!")

from transformers import pipeline

qa = pipeline("question-answering", model="/content/drive/MyDrive/roberta_squad_model", tokenizer=model_name)

qa({"question": "What is the purpose of SQuAD?", "context": "SQuAD is a dataset for question answering tasks."})
