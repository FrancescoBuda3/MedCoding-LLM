from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from multiprocessing import cpu_count
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import unsloth_train
import torch
import re
import random
import os
from datasets import Dataset
import wandb
import pandas as pd
import numpy as np
import random as rand
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import train_on_responses_only


from dotenv import load_dotenv

from huggingface_hub import login
access_token = "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"
login(access_token)
wandb.login(key="fc1d059ed23efd29e58116a6f4b34502eb0bb5cd")

max_seq_length = 16000 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

import json
with open('./data/chat.jsonl') as f:
    data = [json.loads(line) for line in f.readlines()] 

train_dataset = Dataset.from_pandas(pd.DataFrame(data))

def formatting_prompts_func(examples):
    convos = examples['conversation']
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return {"text": texts}

train_dataset = train_dataset.map(formatting_prompts_func, batched = True,)

os.makedirs("./logs", exist_ok=True)
with open("logs/example_prompt.txt", "a") as f:
    f.write(train_dataset[8]["text"])
    f.write("\n")
    f.write("*"*50)

### PARAMS ###
batch_size = 2
acc_steps = 4
lr = 2e-4

run_name = f"Llama-ICD-coder-3B-batch{batch_size}-acc{acc_steps}-lr{lr}"

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = acc_steps,
        warmup_ratio = 0.1,
        #max_steps = 60,
        learning_rate = lr,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 50,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 42,
        output_dir = "outputs",
        num_train_epochs = 3,
        save_strategy="epoch",
        run_name=run_name
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)


# unsloth_train fixes gradient_accumulation_steps
# trainer_stats = trainer.train()
trainer_stats = unsloth_train(trainer)

model.push_to_hub("FrancescoBuda/Llama-ICD-coder-3B")
