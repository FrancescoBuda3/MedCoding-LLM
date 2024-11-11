import pandas as pd
import numpy as np
import gc
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

from huggingface_hub import login
access_token = "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"
login(access_token)

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')


from settings import (
    OUTPUT_DIR,
    DATA_DIR
)




d_icd_diagnoses = pd.read_csv(DATA_DIR + "d_icd_diagnoses.csv.gz", compression="gzip")
d_icd10_diagnoses = d_icd_diagnoses[d_icd_diagnoses.icd_version==10]

goldTranslator = {}
for index, row in d_icd10_diagnoses.iterrows():
    clean = row['long_title'].replace(',', '')
    goldTranslator[clean] = row['long_title']

mimicCodes = set(d_icd10_diagnoses["icd_code"].apply(lambda x: x.replace('.', '')))

orderedCodes = []
with open(DATA_DIR + "2020order.txt", 'r') as f:
    for line in f:
        splitted = line.split()
        if splitted[2] == "1":
            orderedCodes.append(splitted[1])

codes = set(orderedCodes).intersection(mimicCodes)

# DICTIONARY CODE -> DESCRIPTION
cod2lbl = {}
for key, value in zip(d_icd10_diagnoses.icd_code, d_icd10_diagnoses.long_title):
    cod2lbl[key]=value

# DICTIONARY DESCRIPTION -> CODE
lbl2cod = {}
for key, value in zip(d_icd10_diagnoses.long_title, d_icd10_diagnoses.icd_code):
    lbl2cod[key]=value

# FUNCTION TO ASSIGN DESCRIPTION TO CODES
def assign_title(x):
    return [cod2lbl[el.replace('.', '')] for el in x]

# FUNCTION TO ASSIGN CODES TO DESCRIPTION
def assign_codes(x):
    return [lbl2cod[el] for el in x]

del d_icd_diagnoses, d_icd10_diagnoses
gc.collect()

icd10_df = pd.read_feather(DATA_DIR + "mimiciv_icd10.feather")

data = []
with open(OUTPUT_DIR + "testEnts.jsonl", 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

data = df.merge(icd10_df, how="inner", on="note_id")[["note_id", "entities", "_id", "icd10_diag", "raw_text", "icd10_diag_titles"]]


assignedCodes = set([x.replace(".","") for x in data.icd10_diag.explode().dropna().tolist()])
codes = list(codes.union(assignedCodes))

del assignedCodes, df, icd10_df
gc.collect()




df = []
with open(OUTPUT_DIR + "selected.jsonl", 'r') as f:
    for line in f:
        df.append(json.loads(line))

df = pd.DataFrame(df)

data = df.merge(data, how="inner", on="note_id")

print(data["selected"])


sampling_params = SamplingParams(
    n=1,
    temperature=0.0,
    top_p=1.0,
    max_tokens=1024,
    #use_beam_search=False,
)

model = LLM(
    model = "FrancescoBuda/Llama-ICD-coder-1B-merged-2ep", 
    gpu_memory_utilization=.95,
    dtype="auto", 
    enforce_eager=True,
    max_model_len=16000,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("FrancescoBuda/Llama-ICD-coder-1B-merged-2ep")

sys_prompt = "You are an expert in medical coding, specialized in ICD-10 classifications. Based on the medical note provided by the user, identify and assign the most accurate ICD-10 codes from the list of possible ones."

prompts = []
for index, row in data.iterrows():
    input_note = row["raw_text"]
    sel = row["selected"]
    sel_clean = [s.replace(',', '') for s in sel]
    selJoined = '\n- '.join(sel_clean)

    prompt = [
              {"role": "system", "content": sys_prompt},
              {"role": "user", "content": f"Read the following medical note carefully:\n{input_note.strip()}"},
              {"role": "assistant", "content": "Understood, I’ve carefully read the medical note."},
              {"role": "user", "content": f"Now classify the note by selecting only the applicable ICD-10 codes from this list:\n{selJoined}"},
    ]
  
    prompts.append((row["note_id"], prompt))

prompts = [(note_id, tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)) for note_id, prompt in prompts]

with open(OUTPUT_DIR + "tmp.txt", "w") as file:
    file.write(prompts[0][1])

BATCH_SIZE = 16
prompts_batched = [prompts[i:i+BATCH_SIZE] for i in range(0, len(prompts), BATCH_SIZE)]
meanRecall = 0
denum = 0
for id_batch, batch in enumerate(tqdm(prompts_batched, desc="Batches processed")):
  input_prompts = [el[1] for el in batch]
  ids = [el[0] for el in batch]
  outputs = model.generate(input_prompts, sampling_params, use_tqdm=False)
  for i, output in enumerate(outputs):
    generated_text = output.outputs[0].text
    row = data.loc[data["note_id"] == ids[i]]
    sel = set(row["selected"].values[0])
    gold = set(row["icd10_diag_titles"].values[0])
    titles = set()
    splitted = generated_text.split("- ")
    splitted = splitted[1:]
    for term in splitted:
      term = term.replace('\n', '')
      if term in goldTranslator:
        title = goldTranslator[term]
        if title in sel:
            titles.add(term)
    out_dict = {
        "note_id": ids[i],
        "selected": list(titles)
    }
    meanRecall += len(titles.intersection(gold))/len(gold) * 100
    denum += 1
    with open(OUTPUT_DIR + 'defSelected.jsonl', 'a') as f:
        json.dump(out_dict, f, ensure_ascii=False)
        f.write('\n')

print(meanRecall/denum)
    


