# to run this script, you need to have the file mimiciv_icd10.feather in the data folder


import pandas as pd
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import re

from huggingface_hub import login
access_token = "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"
login(access_token)


from settings import (
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
    max_model_len=7000,
    trust_remote_code=True,
)

exampleTerms = "- enlarged right testicle\n- exposure to Brucella\n- edema\n- pain\n- Brucella\n- varicose veins\n- jugular vein engorgement\n- fever\n- febrile syndrome\n- orchiepididymitis\n- osteoarticular pain"
exampleText = "We describe the case of a 37-year-old man with a previously active lifestyle, reporting osteoarticular pain of variable location over the past month and fever in the past week, with peaks (morning and evening) of 40°C in the last 24-48 hours, for which he visited the Emergency Department. Prior to the onset of symptoms, he had been in Extremadura, in a region endemic to brucella, consuming unpasteurized goat milk and cheese from the same livestock. Several cases of brucellosis were reported among the diners. During hospitalization for the study of the febrile syndrome with epidemiological history of possible exposure to Brucella, he developed a case of right orchiepididymitis.\n\nPhysical examination reveals: Temperature: 40.2°C; Blood pressure: 109/68 mmHg; Heart rate: 105 bpm. He is conscious, oriented, sweaty, eupneic, and in good nutritional and hydration status. No adenopathy, goiter, or jugular vein engorgement is palpated in the head and neck, with symmetrical carotid pulses. Cardiac auscultation reveals rhythmic heart sounds without murmurs, rubs, or extra sounds. Pulmonary auscultation shows preservation of vesicular breath sounds. The abdomen is soft, depressible, with no masses or organomegaly. Neurological examination does not detect meningeal signs or focal neurological data. The extremities show no varicose veins or edema. Peripheral pulses are present and symmetrical. Urological examination reveals the right testicle is enlarged, not adherent to the skin, with fluctuation areas and intensely painful on palpation, with loss of the epididymo-testicular boundary and positive transillumination.\n\nAnalytical data show the following results: Blood count: Hb 13.7 g/dl; leukocytes 14,610/mm³ (neutrophils 77%); platelets 206,000/mm³. ESR: 40 mm in the first hour. Coagulation: Prothrombin time 87%; APTT 25.8 seconds. Biochemistry: Glucose 117 mg/dl; urea 29 mg/dl; creatinine 0.9 mg/dl; sodium 136 mEq/l; potassium 3.6 mEq/l; AST 11 U/l; ALT 24 U/l; GGT 34 U/l; alkaline phosphatase 136 U/l; calcium 8.3 mg/dl. Urine: normal sediment.\n\nDuring hospitalization, blood cultures were requested: positive for Brucella, and specific serologies for Brucella: Rose Bengal +++; Coombs test > 1/1280; Brucellacapt > 1/5120. The requested imaging tests (chest X-ray, abdominal ultrasound, cranial CT, transthoracic echocardiogram) did not show significant pathology, except for the testicular ultrasound, which showed thickening of the scrotal sac with a small amount of fluid with septations and an enlarged testicle with small hypoechoic areas inside that could represent microabscesses.\n\nWith the diagnosis of orchiepididymitis secondary to Brucella, symptomatic treatment (antipyretics, anti-inflammatories, rest, and testicular elevation) was initiated, as well as specific antibiotic treatment: Doxycycline 100 mg orally every 12 hours (for 6 weeks) and Streptomycin 1 gram intramuscularly every 24 hours (for 3 weeks). The patient showed significant improvement after one week of hospitalization, and discharge was decided, with the patient completing the antibiotic treatment at home. In successive follow-up consultations, complete resolution of the condition was confirmed."

notes = pd.read_feather(DATA_DIR + "mimiciv_icd10.feather")
split = pd.read_feather("./mimicSplits/mimiciv_icd10/mimiciv_icd10_split.feather")

icd10_train = split[split['split'] == 'train']
icd10_val = split[split['split'] == 'val']
icd10_test = split[split['split'] == 'test']

icd10_train_df = notes[notes['_id'].isin(icd10_train['_id'])].reset_index(drop=True)
notes = icd10_train_df[:5]
notes["entities"] = [[] for _ in range(len(notes))]

prompts = []
for index, row in notes.iterrows():
  prompt = [
    {"role": "system", "content": "you are a medical expert"},
    {"role": "user", "content": "read carefully the following medical note, then i'll tell you what to do.\n\nMedical Note:\n" + row["raw_text"]},
    {"role": "assistant", "content": "Ok, I've read carefully the content of the medical note."},
    {"role": "user", "content": "Extract all the most important medical terms from the note and write them in this format:\n- Term1\n- Term2\n- Term3\n...\n\n. Do not include any additional information.\n\nHere is an example to help you understand the task:\n\nExample Note:\n" + exampleText + "\n\nExample terms:\n"+ exampleTerms }
  ]
  
  prompts.append((row["note_id"], prompt))


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
prompts = [(note_id, tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)) for note_id, prompt in prompts]

BATCH_SIZE = 64
prompts_batched = [prompts[i:i+BATCH_SIZE] for i in range(0, len(prompts), BATCH_SIZE)]

for id_batch, batch in enumerate(tqdm(prompts_batched, desc="Batches processed")):
  input_prompts = [el[1] for el in batch]
  ids = [el[0] for el in batch]
  outputs = llm.generate(input_prompts, sampling_params, use_tqdm=False)
  for i, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    entities = set()
    splitted = generated_text.split("- ")
    for term in splitted:
      term = term.replace('\n', '')
      entities.add(term)
    notes.loc[notes["note_id"] == ids[i], "entities"] = [entities]


notes.to_feather(OUTPUT_DIR + "mimiciv_icd10_entities.feather")
    


