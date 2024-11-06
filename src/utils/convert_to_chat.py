from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from multiprocessing import cpu_count
from trl import SFTTrainer
from tqdm import tqdm
from transformers import TrainingArguments
import torch
import re
import random
from datasets import Dataset
import pandas as pd
import numpy as np
import random as rand
from unsloth.chat_templates import get_chat_template

from huggingface_hub import login
access_token = "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"
login(access_token)

#from src.settings import (
#    OUTPUT_DIR,
#    DATA_DIR
#)

OUTPUT_DIR = "./output/datasets/"
DATA_DIR = "./data/"

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


codes = []
for i in range(0, len(data)):
    if data['only_distractors'][i] == True:
        tmp = list(data['distractors'][i])
    else:
        tmp = list(data['icd10_diag'][i]) + list(data['distractors'][i])
    rand.shuffle(tmp)
    codes.append(tmp)

data['codes'] = codes

data.drop(columns=['distractors', "_id"], inplace=True)
data['responseTitles'] = data['icd10_diag'].map(assign_title)
data['codeTitles'] = data['codes'].map(assign_title)

data['response_string'] = "- " + data['responseTitles'].apply(lambda x: '\n- ' .join(x))
data['code_string'] = "- " + data['codeTitles'].apply(lambda x: '\n- ' .join(x))

data.loc[data["only_distractors"] == True, "response_string"] = ""

data['instruction'] = "You are an expert in medical coding, specialized in ICD-10 classifications. Based on the medical note provided by the user, identify and assign the most accurate ICD-10 codes from the list of possible ones."

conversations = []
for i, item in tqdm(data.iterrows(), total=len(data)):
    sys_prompt  = item["instruction"]
    codes       = item["code_string"]
    input_note  = item["raw_text"]
    output      = item["response_string"]

    conversations.append({
        "id": item["note_id"],
        "conversation": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Read the following medical note carefully:\n{input_note.strip()}"},
            {"role": "assistant", "content": "Understood, I’ve carefully read the medical note."},
            {"role": "user", "content": f"Now classify the note by selecting only the applicable ICD-10 codes from this list:\n{codes}"},
            {"role": "assistant", "content": output},
        ]
    })

import json
for convo in conversations:
    with open("data/chat.jsonl", "a") as f:
        json.dump(convo, f)
        f.write("\n")

# conversations = pd.DataFrame(conversations)

# dataset = Dataset.from_pandas(conversations)

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "unsloth/Llama-3.2-3B",
#     max_seq_length = 16000,
#     dtype = None,
#     load_in_4bit = True,
#     # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
# )

# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template = "llama-3.1",
# )
# convos = dataset['conversation']
# texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]

# for i in range(10):
#     print(texts[i])
#     print("*"*50)
    