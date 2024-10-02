import pandas as pd
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import re
import json

from huggingface_hub import login
access_token = "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"
login(access_token)

from settings import (
    OUTPUT_DIR,
    DATA_DIR
)

notes = pd.read_feather(DATA_DIR + "mimiciv_icd10_entities.feather")

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

goldTranslator = {}

entities = []
with open(OUTPUT_DIR + 'ents.jsonl', 'r') as f:
    for line in f:
        entities.append(json.loads(line))

entities = pd.DataFrame(entities)

prompts = []
for index, row in entities.iterrows():
    note_id = row["note_id"]
    ents = row["entities"]
    golds = notes[notes['note_id'] == note_id]['icd10_diag_titles'].values
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

# stampa il numero massimo di tokens estratti dal tokenizer
max_tokens = 0
for note_id, prompt in prompts:
    tokens = len(tokenizer(prompt).input_ids)
    if tokens > max_tokens:
        max_tokens = tokens

print(max_tokens)

BATCH_SIZE = 32
prompts_batched = [prompts[i:i+BATCH_SIZE] for i in range(0, len(prompts), BATCH_SIZE)]


for id_batch, batch in enumerate(tqdm(prompts_batched, desc="Batches processed")):
    input_prompts = [el[1] for el in batch]
    ids = [el[0] for el in batch]
    outputs = llm.generate(input_prompts, sampling_params, use_tqdm=False)
    for i, output in enumerate(outputs):
        cleanEntities = set(notes.loc[notes["note_id"] == ids[i], "entities"].values)
        golds = set(notes.loc[notes["note_id"] == ids[i], "icd10_diag_titles"].values)
        generated_text = output.outputs[0].text
        splitted = generated_text.split("- ")
        pairs = []
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
                        pairs.append({
                            "term": term,
                            "label": goldTranslator[label]
                        })
        out_dict = {
            "note_id": ids[i],
            "pairs": pairs
        }
        with open(OUTPUT_DIR + 'tmpPairs.jsonl', 'a') as f:
            json.dump(out_dict, f, ensure_ascii=False)
            f.write('\n')

