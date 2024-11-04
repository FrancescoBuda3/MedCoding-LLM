#from unsloth import FastLanguageModel
#from unsloth import is_bfloat16_supported
#from multiprocessing import cpu_count
#from trl import SFTTrainer
#from transformers import TrainingArguments
#import torch
#import re
import random
from datasets import Dataset
import pandas as pd
import numpy as np
import random as rand

from settings import (
    OUTPUT_DIR,
    DATA_DIR
)

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
#fourbit_models = [
#    "unsloth/mistral-7b-v0.3-bnb-4bit",      # New Mistral v3 2x faster!
#    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
#    "unsloth/llama-3-8b-bnb-4bit",           # Llama-3 15 trillion tokens model 2x faster!
#    "unsloth/llama-3-8b-Instruct-bnb-4bit",
#    "unsloth/llama-3-70b-bnb-4bit",
#    "unsloth/Phi-3-mini-4k-instruct",        # Phi-3 2x faster!
#    "unsloth/Phi-3-medium-4k-instruct",
#    "unsloth/mistral-7b-bnb-4bit",
#    "unsloth/gemma-7b-bnb-4bit",             # Gemma 2.2x faster!
#] # More models at https://huggingface.co/unsloth
#
#model, tokenizer = FastLanguageModel.from_pretrained(
#    model_name = "unsloth/llama-3-8b-bnb-4bit",
#    max_seq_length = max_seq_length,
#    dtype = dtype,
#    load_in_4bit = load_in_4bit,
#    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
#)


#model = FastLanguageModel.get_peft_model(
#    model,
#    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
#    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
#                      "gate_proj", "up_proj", "down_proj",],
#    lora_alpha = 16,
#    lora_dropout = 0, # Supports any, but = 0 is optimized
#    bias = "none",    # Supports any, but = "none" is optimized
#    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
#    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
#    random_state = 3407,
#    use_rslora = False,  # We support rank stabilized LoRA
#    loftq_config = None, # And LoftQ
#)
#
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Possible_Codes:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = "FINE"#tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    codes        = examples["code_string"]
    inputs       = examples["raw_text"]
    outputs      = examples["response_string"]
    texts = []
    for instruction,codes, input, output,  in zip(instructions, codes, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, codes, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

# Load the dataset
data = pd.read_feather(DATA_DIR + "mimiciv_icd10.feather")[["note_id", "raw_text"]]
diagnoses = pd.read_csv(DATA_DIR + "d_icd_diagnoses.csv.gz", compression="gzip")
distractors = pd.read_feather(OUTPUT_DIR + "mimiciv_icd10_distractors.feather")

cod2lbl = {}
for key, value in zip(diagnoses.icd_code, diagnoses.long_title):
    cod2lbl[key]=value

def assign_title(x):
    return [cod2lbl[el.replace('.', '')] for el in x]

data = data.merge(distractors, how='inner', on='note_id')
del distractors
import gc
gc.collect()

codes = []
for i in range(0, len(data)):
    tmp = list(data['icd10_diag'][i]) + list(data['distractors'][i])
    rand.shuffle(tmp)
    codes.append(tmp)

data['codes'] = codes

data.drop(columns=['distractors', "note_id", "_id"], inplace=True)
data['responseTitles'] = data['icd10_diag'].map(assign_title)
data['codeTitles'] = data['codes'].map(assign_title)

data['response_string'] = "-" + data['responseTitles'].apply(lambda x: '\n-' .join(x))
data['code_string'] = "-" + data['codeTitles'].apply(lambda x: '\n-' .join(x))

data['instruction'] = "You're a medical coder, Classify the medical note in the input with corresponding ICD-10 codes from the list of the possible ones."

del diagnoses
del cod2lbl
gc.collect()

dataset = Dataset.from_pandas(data)
train_dataset = dataset.map(formatting_prompts_func, batched = True,)

print(train_dataset[3]["text"])


#trainer = SFTTrainer(
#    model = model,
#    tokenizer = tokenizer,
#    train_dataset = train_dataset,
#    dataset_text_field = "text",
#    max_seq_length = max_seq_length,
#    dataset_num_proc = 2,
#    packing = False, # Can make training 5x faster for short sequences.
#    args = TrainingArguments(
#        per_device_train_batch_size = 2,
#        gradient_accumulation_steps = 4,
#        warmup_steps = 5,
#        max_steps = 60,
#        learning_rate = 2e-4,
#        fp16 = not is_bfloat16_supported(),
#        bf16 = is_bfloat16_supported(),
#        logging_steps = 1,
#        optim = "adamw_8bit",
#        weight_decay = 0.01,
#        lr_scheduler_type = "linear",
#        seed = 3407,
#        output_dir = "outputs",
#    ),
#)
#
#trainer_stats = trainer.train()