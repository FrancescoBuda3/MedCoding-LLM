import pandas as pd
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import re
import json
import argparse
import gc
import torch

torch.cuda.empty_cache()
gc.collect()

parser = argparse.ArgumentParser()

parser.add_argument('startIndex', type=int, help="L'indice di partenza")
parser.add_argument('endIndex', type=int, help="L'indice di fine")

args = parser.parse_args()
START_INDEX = args.startIndex
END_INDEX = args.endIndex


from huggingface_hub import login
access_token = "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"
login(access_token)

from settings import (
    OUTPUT_DIR,
    DATA_DIR
)

notes = pd.read_feather(DATA_DIR + "mimiciv_icd10.feather")[["note_id", "raw_text", "icd10_diag_titles"]]
diagnoses = pd.read_csv(DATA_DIR + "d_icd_diagnoses.csv.gz", compression="gzip")

dati = []
with open(OUTPUT_DIR + "tmpPairs.jsonl", 'r') as f:
    for line in f:
        dati.append(json.loads(line))

pairs = pd.DataFrame(dati)
pairs = pairs.groupby('note_id').agg({'pairs': 'sum'}).reset_index()
pairs = pairs[START_INDEX:END_INDEX]
pairs = pairs.merge(notes, how='inner', on='note_id')



goldTranslator = {}
for index, row in diagnoses.iterrows():
    clean = row['long_title'].replace(',', '')
    goldTranslator[clean] = row['long_title']

del diagnoses

# Create a sampling params object.
sampling_params = SamplingParams(
    n=1,
    temperature=0.0,
    top_p=1.0,
    max_tokens=1024,
    use_beam_search=False,
)

# Create an LLM.
llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    gpu_memory_utilization=.95,
    dtype="auto", 
    enforce_eager=True,
    max_model_len=10000,
    trust_remote_code=True,
)

BATCH_SIZE = 32
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

def generate_and_save(prompts):
    prompts = [(note_id, tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)) for note_id, prompt in prompts]
    prompts_batched = [prompts[i:i+BATCH_SIZE] for i in range(0, len(prompts), BATCH_SIZE)]
    for id_batch, batch in enumerate(prompts_batched):
        input_prompts = [el[1] for el in batch]
        ids = [el[0] for el in batch]
        outputs = llm.generate(input_prompts, sampling_params, use_tqdm=False)
        for i, output in enumerate(outputs):
            row = pairs.loc[pairs["note_id"] == ids[i]]
            tmpPairs = []
            for pair in row["pairs"].values[0]:
                term = pair["term"]
                label = pair["label"]
                tmpPairs.append((term, label))
            golds = set(row["icd10_diag_titles"].values[0])
            generated_text = output.outputs[0].text
            splitted = generated_text.split("- ")
            pp = []
            for term in splitted:
                term = term.replace('\n', '')
                if ',' in term:
                    splitted2 = term.split(", ")
                    if len(splitted2) == 3:
                        term = splitted2[0]
                        label = splitted2[1]
                        flag = splitted2[2]
                        if label in goldTranslator:
                            label = goldTranslator[label]
                        else:
                            label = "None"
                        if flag == "YES" and (term, label) in tmpPairs:
                            pp.append({
                                "term": term,
                                "label": label
                            })
            out_dict = {
                "note_id": ids[i],
                "pairs": pp
            }

            with open(OUTPUT_DIR + 'pairs.jsonl', 'a') as f:
                json.dump(out_dict, f, ensure_ascii=False)
                f.write('\n')


prompts = []
for index, row in tqdm(pairs.iterrows(), desc = "validating pairs"):
    tmpPairs = []
    for pair in row["pairs"]:
        term = pair["term"]
        label = pair["label"]
        label = label.replace(',', '')
        tmpPairs.append((term, label))
    tmpPairsItems = ['- '+pair[0]+ ', '+ pair[1]  for pair in tmpPairs]
    n = 10
    groups = [tmpPairsItems[i:i+n] for i in range(0, len(tmpPairsItems), n)]
    for group in groups:
        tmpPairsJoined = '\n'.join(group)
        prompt = [
            {"role": "system", "content": "you are a medical expert"},
            {"role": "user", "content": "Consider the following medical note as context:\n" + row["raw_text"] + "\n\nGiven the following pairs where the first element is a medical term and the second is a label, identify whether there is a medical connection between the term and the label (YES/NO).\n\nPairs:\n" + tmpPairsJoined + "\n\nThe output should follow this format:\n- Term1, Label for Term1, YES\n- Term2, Label for Term2, NO\n- Term3, Label for Term3, MAYBE\n...\n\nDo not include any additional information"}
        ]

        prompts.append((row["note_id"], prompt))
        if len(prompts) >= BATCH_SIZE:
            generate_and_save(prompts)
            prompts = []

generate_and_save(prompts)

del llm

