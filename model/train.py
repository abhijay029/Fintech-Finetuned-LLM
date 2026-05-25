from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    get_peft_model
)

import torch

model_name = "HuggingFaceTB/SmolLM2-135M"

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config = bnb_config, device_map = "auto")

lora_config = LoraConfig(
    r = 16,
    lora_alpha = 32,
    target_modules = ['q_proj', 'v_proj'],
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files = "./data/finance_dataset.json")

def format_data(example):
    text = f"""
    Instruction:
    {example['instruction']}
    Response:
    {example['response']}
    """
    return {"text": text}

dataset = dataset.map(format_data)

def tokenize(example):

    tokens =  tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    
    tokens["labels"] = tokens["input_ids"].copy()
    
    return tokens

tokenized_dataset = dataset.map(tokenize)

training_args = TrainingArguments(
    output_dir = "./fine_tuned",
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    learning_rate = 2e-4,
    num_train_epochs = 3,
    logging_steps = 1,
    save_strategy = "epoch",
    fp16 = True 
)

trainer = Trainer(model = model, args = training_args, train_dataset = tokenized_dataset["train"])

trainer.train()

trainer.model.save_pretrained(
"./fine_tuned_model"
)

tokenizer.save_pretrained(
"./fine_tuned_model"
)

print("Training Completed")