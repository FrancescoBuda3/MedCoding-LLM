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

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

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


matryoshka_dim = 768
sentenceTransformerModel = SentenceTransformer(DATA_DIR + "checkpoint-23900", truncate_dim=matryoshka_dim)


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

# MIMIC-IV SPLITS
split = pd.read_feather("./mimicSplits/mimiciv_icd10/mimiciv_icd10_split.feather")
train = split[split["split"] == "train"]
val = split[split["split"] == "val"]
test = split[split["split"] == "test"]

# MIMIC-IV ELABORATED TABLES
icd10_df = pd.read_feather(DATA_DIR + "mimiciv_icd10.feather")

#icd10_train_df = icd10_df[icd10_df['_id'].isin(train['_id'])].reset_index(drop=True)
#icd10_val_df = icd10_df[icd10_df['_id'].isin(val['_id'])].reset_index(drop=True)
icd10_test_df = icd10_df[icd10_df['_id'].isin(test['_id'])].reset_index(drop=True)

data = []
with open(OUTPUT_DIR + "testEnts.jsonl", 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

data = df.merge(icd10_df, how="inner", on="note_id")[["note_id", "entities", "_id", "icd10_diag", "raw_text", "icd10_diag_titles"]]


assignedCodes = set([x.replace(".","") for x in data.icd10_diag.explode().dropna().tolist()])
codes = list(codes.union(assignedCodes))

del assignedCodes, df, icd10_df, icd10_test_df, split, train, val, test
gc.collect()

data = data.sample(2, random_state=42)

titles = assign_title(codes)
titlesEmbeddings = sentenceTransformerModel.encode(titles)

meanPercentage = 0
meanNumCodes = 0


for index, row in data.iterrows():
  selectedCodes = set()
  targetNames = set(assign_title(list(row["icd10_diag"])))
  entities = row["entities"]

  if(len(entities) > 0):
      entitiesEmbeddings = sentenceTransformerModel.encode(entities)
      similarity = sentenceTransformerModel.similarity(entitiesEmbeddings, titlesEmbeddings)
      maxvals = np.zeros(similarity.shape[1])
      for i in range(similarity.shape[1]):
        colonna = similarity[:, i].cpu().numpy()
        maxvals[i] = np.max(colonna)
    
      maxvals = np.argsort(maxvals)[::-1][:100]
      selectedCodes = set([titles[i] for i in maxvals])
      intersection = targetNames.intersection(selectedCodes)
     
      
      percentage = (len(intersection) / len(targetNames)) * 100
      print(f"Percentage of targetNames in selectedCodes: {percentage:.2f}%")
      print(f"num of selected codes: {len(selectedCodes)}")

      out_dict = {
        "note_id": row["note_id"],
        "selected": list(selectedCodes)
      }
      

  else:
      percentage = 0
      out_dict = {
        "note_id": row["note_id"],
        "selected": []
      }
  meanPercentage += percentage
  meanNumCodes += len(selectedCodes)
  with open(OUTPUT_DIR + 'selected.jsonl', 'a') as f:
        json.dump(out_dict, f, ensure_ascii=False)
        f.write('\n')

print(f"mean percentage: {meanPercentage / len(data)}")
print(f"mean num of selected codes: {meanNumCodes / len(data)}")

df = []
with open(OUTPUT_DIR + "selected.jsonl", 'r') as f:
    for line in f:
        df.append(json.loads(line))

df = pd.DataFrame(df)

data = df.merge(data, how="inner", on="note_id")

print(data["selected"])

from unsloth import FastLanguageModel

sampling_params = SamplingParams(
    n=1,
    temperature=0.0,
    top_p=1.0,
    max_tokens=1024,
    #use_beam_search=False,
)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "FrancescoBuda/Llama-ICD-coder-1B-merged", 
    max_seq_length = 16000,
    dtype = None,
    load_in_4bit = True,
)

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

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

prompts = [(note_id, tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)) for note_id, prompt in prompts]

with open(OUTPUT_DIR + "tmp.txt", "w") as file:
    file.write(prompts[0][1])

""" BATCH_SIZE = 8
prompts_batched = [prompts[i:i+BATCH_SIZE] for i in range(0, len(prompts), BATCH_SIZE)]

for id_batch, batch in enumerate(tqdm(prompts_batched, desc="Batches processed")):
  input_prompts = [el[1] for el in batch]
  ids = [el[0] for el in batch]
  outputs = model.generate(input_prompts, sampling_params, use_tqdm=False)
  for i, output in enumerate(outputs):
    generated_text = output.outputs[0].text
    row = data.loc[data["note_id"] == ids[i]]
    sel = row["selected"]
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
        "note_id": row["note_id"],
        "selected": list(titles)
    }
    with open(OUTPUT_DIR + 'defSelected.jsonl', 'a') as f:
        json.dump(out_dict, f, ensure_ascii=False)
        f.write('\n') """


