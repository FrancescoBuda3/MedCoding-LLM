
import json
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd
from unsloth.chat_templates import get_chat_template


tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B")

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

from settings import (
    OUTPUT_DIR,
    DATA_DIR
)


with open('./data/chat.jsonl') as f:
    data = [json.loads(line) for line in f.readlines()]
 
train_dataset = Dataset.from_pandas(pd.DataFrame(data))
 
def formatting_prompts_func(examples):
    convos = examples['conversation']
    texts = [tokenizer.apply_chat_template(convo, tokenize = True, add_generation_prompt = False) for convo in convos]
    return {"text": texts}
 
train_dataset = train_dataset.map(formatting_prompts_func, batched = True,)

max_seq_length = 0
mean_seq_length = 0
num_mag_8k = 0
lunghezze = []
for text in train_dataset['text']:
    lunghezza = len(text)
    lunghezze.append(lunghezza)
    max_seq_length = max(max_seq_length, lunghezza)
    mean_seq_length += lunghezza
    if lunghezza > 8000 :
        num_mag_8k += 1

mean_seq_length /= len(train_dataset['text'])
print(f"Max sequence length: {max_seq_length}")
print(f"Mean sequence length: {mean_seq_length}")
print(f"Num > 8k: {num_mag_8k}")

print(train_dataset.features)



df = train_dataset.to_pandas()
df["lunghezza"] = lunghezze
df = df[['id', 'lunghezza']]

# salva il dataset in csv

df.to_csv(OUTPUT_DIR + 'lenghts.csv', index=False)