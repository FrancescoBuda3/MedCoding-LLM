import pandas as pd
import numpy as np
import random as rand
rand.seed(42)

from settings import (
    OUTPUT_DIR,
    DATA_DIR
)

data = pd.read_feather(DATA_DIR + "mimiciv_icd10.feather")[["note_id", "_id", "icd10_diag"]]
diagnoses = pd.read_csv(DATA_DIR + "d_icd_diagnoses.csv.gz", compression="gzip")
mimic_split = pd.read_feather("./mimicSplits/mimiciv_icd10/mimiciv_icd10_split.feather")
mimic_train = mimic_split[mimic_split['split'] == 'train']
mimic_train_df = data[data['_id'].isin(mimic_train['_id'])]

orderedCodes = []
with open(DATA_DIR + "2020order.txt", 'r') as f:
    for line in f:
        splitted = line.split()
        if splitted[2] == "1":
            orderedCodes.append(splitted[1])

codes = set(orderedCodes)

cod2lbl = {}
for key, value in zip(diagnoses.icd_code, diagnoses.long_title):
    cod2lbl[key]=value

# REMOVE DOTS FROM CODES
#for i in range(0, len(mimic_train_df)):
#    tmp = mimic_train_df.icd10_diag.iloc[i]
#    tmp = [x.replace('.', '') for x in tmp]
#    mimic_train_df.loc[i, 'icd10_diag'] = tmp

mimic_train_df['icd10_diag'] = mimic_train_df['icd10_diag'].apply(lambda lst: [x.replace('.', '') for x in lst])


def create_distractors(x):
    target_set = set(x).intersection(codes)
    num_targets = len(target_set)
    m = 250 - num_targets
    num_total_distractors = rand.randint(50, m)
    num_near_distractors = rand.randint(0, num_total_distractors)
    num_random_distractors = num_total_distractors - num_near_distractors

    similar_codes = set()
    for code in target_set:
        ind = orderedCodes.index(code)
        start = max(0, ind - 10)
        end = min(len(orderedCodes), ind + 10)
        similar_codes.update(orderedCodes[start:end])
    
    similar_codes = similar_codes - target_set
    other_codes = codes - target_set - similar_codes

    similar_codes = list(similar_codes)
    other_codes = list(other_codes)
    num_near_distractors = min(num_near_distractors, len(similar_codes))
    num_random_distractors = num_total_distractors - num_near_distractors
    distractors = rand.sample(similar_codes, num_near_distractors) + rand.sample(other_codes, num_random_distractors)
       
    return distractors

from tqdm import tqdm

# Aggiungi tqdm alla funzione apply per mostrare il caricamento
tqdm.pandas()
mimic_train_df['distractors'] = mimic_train_df['icd10_diag'].progress_apply(create_distractors)
mimic_train_df['only_distractors'] = np.random.choice([False, True], mimic_train_df.shape[0], p=[0.95, 0.05])

mimic_train_df.reset_index(drop=True).to_feather(OUTPUT_DIR + "mimiciv_icd10_distractors.feather")

