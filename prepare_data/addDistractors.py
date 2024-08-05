import pandas as pd
import numpy as np
import random as rand

from src.settings import (
    DOWNLOAD_DIRECTORY_MIMICIV,
    DOWNLOAD_DIRECTORY_MIMICIV_NOTE,
    ICD9_ELABORATED_PATH,
    ICD10_ELABORATED_PATH
)


icd9 = pd.read_feather(ICD9_ELABORATED_PATH)
icd10 = pd.read_feather(ICD10_ELABORATED_PATH)

# ICD10 DIAGNOSIS CODES
d_icd_diagnoses = pd.read_csv(DOWNLOAD_DIRECTORY_MIMICIV + "hosp/d_icd_diagnoses.csv.gz", compression="gzip")
d_icd10_diagnoses = d_icd_diagnoses[d_icd_diagnoses.icd_version==10]

# ICD10 PROCEDURE CODES
d_icd_procedures = pd.read_csv(DOWNLOAD_DIRECTORY_MIMICIV + "hosp/d_icd_procedures.csv.gz", compression="gzip")
d_icd10_procedures = d_icd_procedures[d_icd_procedures.icd_version==10]

df_icd10_total = pd.concat([d_icd10_diagnoses, d_icd10_procedures], axis = 0)

# DICTIONARY CODE -> DESCRIPTION
cod2lbl = {}
for key, value in zip(df_icd10_total.icd_code, df_icd10_total.long_title):
    cod2lbl[key]=value

# FUNCTION TO ASSIGN DESCRIPTION TO CODES
def assign_title(x):
    return [cod2lbl[el.replace('.', '')] for el in x]

icd10_split = pd.read_feather("../files/mimiciv_icd10/mimiciv_icd10_split.feather")

# THE SPLITS
icd10_train = icd10_split[icd10_split['split'] == 'train']
icd10_val = icd10_split[icd10_split['split'] == 'val']
icd10_test = icd10_split[icd10_split['split'] == 'test']

icd10_train_df = icd10[icd10['_id'].isin(icd10_train['_id'])]

# RANDOM SAMPLE OF 10000 FOR TESTING
icd10_prova = icd10_train_df.sample(10000, random_state=42).reset_index(drop=True)

# REMOVE DOTS FROM CODES
for i in range(0, len(icd10_prova)):
    tmp = icd10_prova['target'][i]
    for j in range(0, len(tmp)):
        tmp[j] = tmp[j].replace('.', '')


# FUNCTION TO CREATE DISTRACTORS
def create_distractors(x):
    target_set = set(x)
    num_targets = len(target_set)
    target_list = list(x)
    num_distractors = 0
    while num_distractors == 0:
        num_distractors = round(rand.uniform(0.25, 2) * num_targets)
        
    available_codes = set(df_icd10_total["icd_code"]) - target_set
    available_codes_list = list(available_codes)
    distractors = rand.sample(available_codes_list, num_distractors)
        
    return distractors

# ADDING DISTRACTORS TO THE DATAFRAME
icd10_prova['distractors'] = icd10_prova['target'].apply(create_distractors)

# ASSERT THAT NO TARGET IS IN THE DISTRACTORS
for i in range(0, len(icd10_prova)):
    target_set = set(icd10_prova['target'][i])
    distractors_set = set(icd10_prova['distractors'][i])
    assert len(target_set.intersection(distractors_set)) == 0


# CLONING THE TARGET COLUMN IN THE RESPONSE COLUMN
icd10_prova["response"] = icd10_prova["target"]

# EXTRACTING 5% OF THE SAMPLE TO CREATE ROWS WITH ONLY DISTRACTORS
onlyDistractors = icd10_prova.sample(frac=0.05, random_state=42).reset_index(drop=True)

onlyDistractors['distractors'] = onlyDistractors['target'].apply(create_distractors)

# EMPTYING THE RESPONSE COLUMN
onlyDistractors["response"] = onlyDistractors["response"].apply(lambda x: [])

# CONCATENATING THE TWO DATAFRAMES AND SHAFFLING THE ROWS
final_df = pd.concat([icd10_prova, onlyDistractors], axis = 0)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

final_df.to_csv("../output/datasets/test_with_distractors.csv", index=False)