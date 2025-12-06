!pip install -q transformers datasets peft accelerate sentencepiece bitsandbytes

from datasets import load_dataset

#load dataset
dataset = load_dataset("squad")
train_data = dataset["train"].select(range(3000))        
val_data   = dataset["validation"].select(range(500))    

#mistralai has the same GPT architecture as that of GPT-3/4
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

#load tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def format_example(example):
    context = example["context"]
    question = example["question"]
    answer = example["answers"]["text"][0]

    prompt = f"""### Instruction:
Answer the question using the context.

### Context:
{context}

### Question:
{question}

### Answer:
"""

    return {
        "input_text": prompt,
        "target_text": answer,
    }

train_data = train_data.map(format_example)
val_data   = val_data.map(format_example)

def tokenize(batch):
    return tokenizer(
        batch["input_text"],
        text_target=batch["target_text"],
        max_length=256,   
        padding="max_length",
        truncation=True,
    )

train_tokenized = train_data.map(tokenize, batched=True, remove_columns=train_data.column_names)
val_tokenized   = val_data.map(tokenize, batched=True, remove_columns=val_data.column_names)

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,        
    device_map="auto",
)

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                      
    lora_alpha=16,            
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./mistral-squad-fast",
    per_device_train_batch_size=2,     
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,     
    learning_rate=2e-4,
    num_train_epochs=1,
    fp16=True,
    logging_steps=20,
)

#create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
)

#train the model
trainer.train()

#test
context = "Transformers are neural networks that use self attention to process sequences."
question = "What do transformers use to process sequences?"

print(answer_question(context, question))

