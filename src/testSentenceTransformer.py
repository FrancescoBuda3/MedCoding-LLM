import pandas as pd
import numpy as np
import re
import gc

from settings import (
    OUTPUT_DIR,
    DATA_DIR
)


d_icd_diagnoses = pd.read_csv(DATA_DIR + "d_icd_diagnoses.csv.gz", compression="gzip")
d_icd10_diagnoses = d_icd_diagnoses[d_icd_diagnoses.icd_version==10]

mimicCodes = set(d_icd10_diagnoses["icd_code"].apply(lambda x: x.replace('.', '')))

def read_order(file_path):
    dati = []
    with open(file_path, 'r') as file:
        for line in file:
            splitted = list(filter(None, line.split(" ")))[1:3]
            dati.append((splitted[0], True if splitted[1] == '1' else False))
    df = pd.DataFrame(dati, columns=['Codice', 'Flag'])
    return df

file_path = DATA_DIR + '2020order.txt'
assignableCodes = read_order(file_path)
assignableCodes = set(assignableCodes[assignableCodes["Flag"] == True]["Codice"])
filtered_mimicCodes = mimicCodes.intersection(assignableCodes)
codes = list(filtered_mimicCodes)





def preprocess_text(
    text: str,
    lower: bool = True,
    remove_special_characters_mullenbach: bool = True,
    remove_special_characters: bool = False,
    remove_digits: bool = True,
    remove_accents: bool = False,
    remove_brackets: bool = False,
    convert_danish_characters: bool = False
) -> str:
    if lower:
        text = text.lower()
    if convert_danish_characters:
        text = re.sub("å", "aa", text)
        text = re.sub("æ", "ae", text)
        text = re.sub("ø", "oe", text)
    if remove_accents:
        text = re.sub("é|è|ê", "e", text)
        text = re.sub("á|à|â", "a", text)
        text = re.sub("ô|ó|ò", "o", text)
    if remove_brackets:
        text = re.sub("\[[^]]*\]", "", text)
    if remove_special_characters:
        text = re.sub("\n|/|-", " ", text)
        text = re.sub("[^a-zA-Z0-9 .]", "", text)  # Mantiene i punti
    if remove_special_characters_mullenbach:
        text = re.sub("[^A-Za-z0-9.]+", " ", text)  # Mantiene i punti
    if remove_digits:
        text = re.sub("(\s\d+)+\s", " ", text)

    text = re.sub("\s+", " ", text)
    text = text.strip()

    return text

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


del d_icd_diagnoses, assignableCodes, filtered_mimicCodes

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

icd10_df = icd10_test_df

test = icd10_df.sample(20, random_state=42)

del icd10_test_df, icd10_df


gc.collect()

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import re

from huggingface_hub import login
access_token = "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"
login(access_token)


# Create a sampling params object.
sampling_params = SamplingParams(
    n=1,
    temperature=0.0,
    top_p=1.0,
    max_tokens=1024#,
    #use_beam_search=False,
)

# Create an LLM.
llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct", # switch to "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gpu_memory_utilization=.95,
    dtype="auto", # set to "auto" with L4 GPU
    enforce_eager=True,
    max_model_len=15000,
    trust_remote_code=True,
)

exampleTerms = "- enlarged right testicle\n- exposure to Brucella\n- edema\n- pain\n- Brucella\n- varicose veins\n- jugular vein engorgement\n- fever\n- febrile syndrome\n- orchiepididymitis\n- osteoarticular pain"
exampleText = "We describe the case of a 37-year-old man with a previously active lifestyle, reporting osteoarticular pain of variable location over the past month and fever in the past week, with peaks (morning and evening) of 40°C in the last 24-48 hours, for which he visited the Emergency Department. Prior to the onset of symptoms, he had been in Extremadura, in a region endemic to brucella, consuming unpasteurized goat milk and cheese from the same livestock. Several cases of brucellosis were reported among the diners. During hospitalization for the study of the febrile syndrome with epidemiological history of possible exposure to Brucella, he developed a case of right orchiepididymitis.\n\nPhysical examination reveals: Temperature: 40.2°C; Blood pressure: 109/68 mmHg; Heart rate: 105 bpm. He is conscious, oriented, sweaty, eupneic, and in good nutritional and hydration status. No adenopathy, goiter, or jugular vein engorgement is palpated in the head and neck, with symmetrical carotid pulses. Cardiac auscultation reveals rhythmic heart sounds without murmurs, rubs, or extra sounds. Pulmonary auscultation shows preservation of vesicular breath sounds. The abdomen is soft, depressible, with no masses or organomegaly. Neurological examination does not detect meningeal signs or focal neurological data. The extremities show no varicose veins or edema. Peripheral pulses are present and symmetrical. Urological examination reveals the right testicle is enlarged, not adherent to the skin, with fluctuation areas and intensely painful on palpation, with loss of the epididymo-testicular boundary and positive transillumination.\n\nAnalytical data show the following results: Blood count: Hb 13.7 g/dl; leukocytes 14,610/mm³ (neutrophils 77%); platelets 206,000/mm³. ESR: 40 mm in the first hour. Coagulation: Prothrombin time 87%; APTT 25.8 seconds. Biochemistry: Glucose 117 mg/dl; urea 29 mg/dl; creatinine 0.9 mg/dl; sodium 136 mEq/l; potassium 3.6 mEq/l; AST 11 U/l; ALT 24 U/l; GGT 34 U/l; alkaline phosphatase 136 U/l; calcium 8.3 mg/dl. Urine: normal sediment.\n\nDuring hospitalization, blood cultures were requested: positive for Brucella, and specific serologies for Brucella: Rose Bengal +++; Coombs test > 1/1280; Brucellacapt > 1/5120. The requested imaging tests (chest X-ray, abdominal ultrasound, cranial CT, transthoracic echocardiogram) did not show significant pathology, except for the testicular ultrasound, which showed thickening of the scrotal sac with a small amount of fluid with septations and an enlarged testicle with small hypoechoic areas inside that could represent microabscesses.\n\nWith the diagnosis of orchiepididymitis secondary to Brucella, symptomatic treatment (antipyretics, anti-inflammatories, rest, and testicular elevation) was initiated, as well as specific antibiotic treatment: Doxycycline 100 mg orally every 12 hours (for 6 weeks) and Streptomycin 1 gram intramuscularly every 24 hours (for 3 weeks). The patient showed significant improvement after one week of hospitalization, and discharge was decided, with the patient completing the antibiotic treatment at home. In successive follow-up consultations, complete resolution of the condition was confirmed."

def get_llama_entities(text):
  prompts = []
  prompt = [
    {"role": "system", "content": "you are a medical expert"},
    {"role": "user", "content": "read carefully the following medical note, then i'll tell you what to do.\n\nMedical Note:\n\"" + text + "\""},
    {"role": "assistant", "content": "Ok, I've read carefully the content of the medical note."},
    {"role": "user", "content": "Extract all the most important medical terms from the note and write them in this format:\n- Term1\n- Term2\n- Term3\n...\n\n. Do not include any additional text.\n\nHere is an example to help you understand the task:\n\nExample Note:\n\"" + exampleText + "\"\n\nExample terms:\n"+ exampleTerms }
  ]
  prompts.append(prompt)

  tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct") # switch to "meta-llama/Meta-Llama-3.1-8B-Instruct"
  prompts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts]

  BATCH_SIZE = 1 # number of prompts to process simultaneously (need to speed up generation)
  prompts_batched = [prompts[i:i+BATCH_SIZE] for i in range(0, len(prompts), BATCH_SIZE)]

  for id_batch, batch in enumerate(tqdm(prompts_batched, desc="Batches processed")):
    outputs = llm.generate(batch, sampling_params, use_tqdm=False)

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        terms = generated_text.split("\n- ")
        #print(terms)
        return terms

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
#model = SentenceTransformer('alecocc/icd10-hard-negatives')
matryoshka_dim = 768
model = SentenceTransformer(DATA_DIR + "checkpoint-23900", truncate_dim=matryoshka_dim)

titles = [cod2lbl[x] for x in codes]
embeddings2 = model.encode(titles)

meanPercentage = 0
meanNumCodes = 0


for index, row in test.iterrows():
  selectedCodes = set()
  targetNames = set(assign_title(list(row["icd10_diag"])))
  note = row["raw_text"]
  #sentences = textToSentences(note)



  entities = get_llama_entities(note)

  #noteEntities = set(nlp(note).ents)
  #for entity in noteEntities:
  # ents = entity._.kb_ents
  # for umls_ent in ents:
  #   x = linker.kb.cui_to_entity[umls_ent[0]]
  #   name = x.canonical_name
  #   entities.add(name)
  #   synonyms = x.aliases
     #for synonym in synonyms:
      #entities.add(synonym)


  #for sentence in sentences:
  #    sentenceEntities = set(nlp(sentence).ents)
  #    for word in sentence.split():
  #      sentenceEntities.update(set(nlp(word).ents))
  #    for entity in sentenceEntities:
  #      ents = entity._.kb_ents
  #      for umls_ent in ents:
  #        x = linker.kb.cui_to_entity[umls_ent[0]]
  #        name = x.canonical_name
  #        entities.add(name)
  #
  #        #synonyms = x.aliases
  #        #for synonym in synonyms:
  #        #  entities.add(synonym)

  entities = list(entities)
  if(len(entities) > 0):
      embeddings1 = model.encode(entities)
    
      similarity = model.similarity(embeddings1, embeddings2)
    
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
  else:
      percentage = 0
  meanPercentage += percentage
  meanNumCodes += len(selectedCodes)

print(f"mean percentage: {meanPercentage / len(test)}")
print(f"mean num of selected codes: {meanNumCodes / len(test)}")