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

assignedCodes = set([x.replace(".","") for x in icd10_df.icd10_diag.explode().dropna().tolist()])
codes = list(assignedCodes)
print(len(codes))

data = []
with open(OUTPUT_DIR + "testEnts.jsonl", 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

data = df.merge(icd10_df, how="inner", on="note_id")[["note_id", "entities", "_id", "icd10_diag", "raw_text", "icd10_diag_titles"]]


#assignedCodes = set([x.replace(".","") for x in data.icd10_diag.explode().dropna().tolist()])
#codes = list(codes.union(assignedCodes))

del assignedCodes, df, icd10_df, icd10_test_df, split, train, val, test
gc.collect()

data = data.sample(1000, random_state=42)

titles = list(set(assign_title(codes)))
titlesEmbeddings = sentenceTransformerModel.encode(titles)



recall50 = 0
recall100 = 0
recall200 = 0
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
    
      maxvals200 = np.argsort(maxvals)[::-1][:200]
      selectedCodes = set([titles[i] for i in maxvals200])
      intersection = targetNames.intersection(selectedCodes)
      percentage = (len(intersection) / len(targetNames)) * 100
      recall200 += percentage

      maxvals100 = np.argsort(maxvals)[::-1][:100]
      selectedCodes = set([titles[i] for i in maxvals100])
      intersection = targetNames.intersection(selectedCodes)
      percentage = (len(intersection) / len(targetNames)) * 100
      recall100 += percentage

      maxvals50 = np.argsort(maxvals)[::-1][:50]
      selectedCodes = set([titles[i] for i in maxvals50])
      intersection = targetNames.intersection(selectedCodes)
      percentage = (len(intersection) / len(targetNames)) * 100
      recall50 += percentage

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
  with open(OUTPUT_DIR + 'selected.jsonl', 'a') as f:
        json.dump(out_dict, f, ensure_ascii=False)
        f.write('\n')

print(f"mean recall 50: {recall50 / len(data)}")
print(f"mean recall 100: {recall100 / len(data)}")
print(f"mean recall 200: {recall200 / len(data)}")


