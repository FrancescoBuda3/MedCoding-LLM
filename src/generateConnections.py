# to run this script, you need to have the file mimiciv_icd10_entities.feather in the data folder
# this script must be reviewed and tested


import pandas as pd
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import re

from huggingface_hub import login
access_token = "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"
login(access_token)

from src.settings import (
    OUTPUT_DIR,
    DATA_DIR
)

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

data = pd.read_feather(DATA_DIR + "mimiciv_icd10_entities.feather")

data["tmpPairs"] = [set() for _ in range(len(data))]
data["defPairs"] = [set() for _ in range(len(data))]

goldTranslator = {}

prompts = []
for index, row in data.iterrows():
    entities = row["entities"]
    golds = row["icd10_diag_titles"]
    golds_clean = []
    for gold in golds:
        clean = gold.replace(',','')
        if clean not in goldTranslator:
            goldTranslator[clean] = gold
        golds_clean.append(clean)
    goldsJoined = '\n- '.join(golds_clean)    
    n = 10
    groups = [entities[i:i+n] for i in range(0, len(entities), n)]

    for group in groups:
        groupJoined = '\n'.join(group)
        prompt = [
            {"role": "system", "content": "you are a medical expert"},
            {"role": "user", "content": "read carefully the following medical note, then i'll tell you what to do.\n\nMedical Note:\n" + row["raw_text"]},
            {"role": "assistant", "content": "Ok, I've read carefully the content of the medical note."},
            {"role": "user", "content": "These are the medical terms extracted from the note. \n\nTerms:\n"+ groupJoined +"\n\nAssign to each term one of the following label if there is a strong medical connection between them, otherwise just return \"None\": Labels:\n- "+ goldsJoined +"\n\nReturn each pair as a separate item in the following format:\n- term1, label for term1\n- term2, None\n- term3, label for term3\n...\n\nDo not include any additional information."}
        ]

        prompts.append((row["note_id"], prompt))
    
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
prompts = [(note_id, tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)) for note_id, prompt in prompts]

BATCH_SIZE = 32
prompts_batched = [prompts[i:i+BATCH_SIZE] for i in range(0, len(prompts), BATCH_SIZE)]

for id_batch, batch in enumerate(tqdm(prompts_batched, desc="Batches processed")):
    input_prompts = [el[1] for el in batch]
    ids = [el[0] for el in batch]
    outputs = llm.generate(input_prompts, sampling_params, use_tqdm=False)
    for i, output in enumerate(outputs):
        cleanEntities = set(data.loc[data["note_id"] == ids[i], "entities"].values)
        golds = set(data.loc[data["note_id"] == ids[i], "icd10_diag_titles"].values)
        generated_text = output.outputs[0].text
        splitted = generated_text.split("- ")
        for term in splitted:
            term = term.replace('\n', '')
            if ',' in term:
                splitted2 = term.split(", ")
                if (len(splitted2) == 2):
                    term = splitted2[0]
                    label = splitted2[1]
                    if label in goldTranslator:
                        label = goldTranslator[label]
                    else:
                        label = "None"
                    if label != "None" and term in cleanEntities:
                        data.loc[data["note_id"] == ids[i], "tmpPairs"].add((term, label))


prompts = []
for index, row in data.iterrows():
    tmpPairsItems = ['- '+pair[0]+ ', '+ pair[1]  for pair in row["tmpPairs"]]
    groups = [tmpPairsItems[i:i+n] for i in range(0, len(tmpPairsItems), n)]
    for group in groups:
        tmpPairsJoined = '\n'.join(group)
        prompt = [
            {"role": "system", "content": "you are a medical expert"},
            {"role": "user", "content": "Consider the following medical note as context:\n" + row["raw_text"] + "\n\nGiven the following pairs where the first element is a medical term and the second is a label, identify whether there is a medical connection between the term and the label (YES/NO).\n\nPairs:\n" + tmpPairsJoined + "\n\nThe output should follow this format:\n- Term1, Label for Term1, YES\n- Term2, Label for Term2, NO\n- Term3, Label for Term3, MAYBE\n...\n\nDo not include any additional information"}
        ]

        prompts.append(row["note_id"], prompt)
    
prompts = [(note_id, tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)) for note_id, prompt in prompts] 
prompts_batched = [prompts[i:i+BATCH_SIZE] for i in range(0, len(prompts), BATCH_SIZE)]

for id_batch, batch in enumerate(tqdm(prompts_batched, desc="Batches processed")):
    input_prompts = [el[1] for el in batch]
    ids = [el[0] for el in batch]
    outputs = llm.generate(input_prompts, sampling_params, use_tqdm=False)
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        splitted = generated_text.split("- ")
        for term in splitted:
            term = term.replace('\n', '')
            if ',' in term:
                splitted2 = term.split(", ")
                if len(splitted2) == 3:
                    term = splitted2[0]
                    label = splitted2[1]
                    flag = splitted2[2]
                    if flag == "YES" and (term, label) in data.loc[data["note_id"] == ids[i], "tmpPairs"]:
                        data.loc[data["note_id"] == ids[i], "defPairs"].add((term, goldTranslator[label]))


data.to_feather(OUTPUT_DIR + "mimiciv_icd10_pairs.feather")