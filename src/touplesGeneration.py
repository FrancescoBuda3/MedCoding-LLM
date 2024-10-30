from settings import (
    OUTPUT_DIR,
    DATA_DIR
)

import simple_icd_10_cm as cm
import pandas as pd
import numpy as np
import json
import random
import re

diagnoses = pd.read_csv(DATA_DIR + "d_icd_diagnoses.csv.gz", compression="gzip")
diagnoses = diagnoses[diagnoses["icd_version"] == 10]

cod2ents = {}
desc2cod = {}
cod2desc = {}

for index, row in diagnoses.iterrows():
    cod2ents[row["icd_code"]] = set()
    desc2cod[row["long_title"]] = row["icd_code"]
    cod2desc[row["icd_code"]] = row["long_title"]




pp = []
data = {}
with open(OUTPUT_DIR + "pairs.jsonl", 'r') as f:
    for line in f:
        line = json.loads(line)
        id = line["note_id"]
        pairs = line["pairs"]
        pairsList = []
        for pair in pairs:
            if pair["label"] in desc2cod:
                cod2ents[desc2cod[pair["label"]]].add(pair["term"].upper())
                pp.append((pair["term"], desc2cod[pair["label"]]))
                pairsList.append((pair["term"], desc2cod[pair["label"]]))
        if id not in data:
            data[id] = []
        data[id] += pairsList

pairs = pp

tot = list(cod2desc.keys())


def get_cousines(code):
    parent = cm.get_parent(code)
    grandpa = cm.get_parent(parent)
    cousines = [cm.get_children(x) for x in cm.get_children(grandpa) if x != parent]
    cousines = [item for sublist in cousines for item in sublist]
    cousines = [x for x in cousines if cm.is_leaf(x)]
    cousines = [x.replace(".", "") for x in cousines]
    random.shuffle(cousines)
    return cousines

orderedCodes = []
with open(DATA_DIR + "2020order.txt", 'r') as f:
    for line in f:
        splitted = line.split()
        if splitted[2] == "1":
            orderedCodes.append(splitted[1])



def read_range(range):
    equivalentCodes = []
    splitted = range.split("-")
    splitted = [x.replace(" ", "") for x in splitted]
    if len(splitted) == 2 and len(splitted[0]) != 0 and len(splitted[1]) != 0:
        a = orderedCodes.index([x for x in orderedCodes if x.startswith(splitted[0].replace(".", ""))][0])
        b = orderedCodes.index([x for x in orderedCodes if x.startswith(splitted[1].replace(".", ""))][-1])
        equivalentCodes.extend(orderedCodes[a:b+1])
    return equivalentCodes

def read_family(family):
    equivalentCodes = []
    splitted = family.split("-")
    splitted = [x.replace(" ", "") for x in splitted]
    if len(splitted) == 2 and len(splitted[0]) != 0 and len(splitted[1]) == 0:
        equivalentCodes.extend([x.replace(".","") for x in cm.get_descendants(splitted[0].replace(".",""))])
    return [x.replace(".","") for x in equivalentCodes]

def read_code(code):
    splitted = code.split("-")
    splitted = [x.replace(" ", "") for x in splitted]
    equivalentCodes = []
    if len(splitted) == 1:
        descendant = [x.replace(".","") for x in cm.get_descendants(splitted[0].replace(".",""))]
        if len(descendant) > 0:
            equivalentCodes.extend(descendant)
        else:
            equivalentCodes.append(splitted[0].replace(".",""))
    return [x.replace(".","") for x in equivalentCodes]

def read_range_family_code(s):
    rangePattern = r"[A-Z][A-Z0-9]*(\.[A-Z0-9]+)?-[A-Z][A-Z0-9]*(\.[A-Z0-9]+)?"
    familyPattern = r"[A-Z][A-Z0-9]*(\.)?([A-Z0-9]+)?-"
    codePattern = r"^[^-]*$"
    equivalentCodes = []

    if re.match(rangePattern, s):
        equivalentCodes = read_range(s)
    elif re.match(familyPattern, s):
        equivalentCodes = read_family(s)
    elif re.match(codePattern, s):
        equivalentCodes = read_code(s)
    
    return [x.replace(".","") for x in equivalentCodes]
    


def read_with(w):
    equivalentCodes = []
    if "with" in w:
        splitted = w.split("with")
        prefixcodes = splitted[0].split(",")
        prefixcodes = [x.replace(" ", "") for x in prefixcodes]
        suffixcodes = splitted[1].split(",")
        suffixcodes = [x.replace(" ", "").replace(".", "") for x in suffixcodes]

        pre = []
        for p in prefixcodes:
            pre.extend(read_range_family_code(p))
        
        for p in pre:
            p2 = p[3:]
            for s in suffixcodes:
                if p2.startswith(s):
                    equivalentCodes.append(p)
    return [x.replace(".","") for x in equivalentCodes]


        
def get_excludes(code):
    equivalentCodes = []
    if cm.is_valid_item(code):
        negatives = set(cm.get_excludes1(code) + cm.get_excludes2(code))
        for c in negatives:
            if not cm.is_valid_item(c):
                if not c.endswith(")"):
                    continue
                else:
                    match = re.search(r'\(([^()]*)\)(?!.*\([^()]*\))',  c)
                    if match:
                        extracted_codes = [match.group(1)]
                    else:
                        extracted_codes = []
                    if len(extracted_codes) > 0:
                        extracted_code = extracted_codes[0]
                        if "with final characters" in extracted_code:
                            equivalentCodes.extend([])
                        if "with" in extracted_code:
                            equivalentCodes.extend(read_with(extracted_code))
                        else:
                            splitted = extracted_code.split(",")
                            splitted = [x.replace(" ", "") for x in splitted]
                            for s in splitted:
                                if s != "":
                                    equivalentCodes.extend(read_range_family_code(s))
            else:
                equivalentCodes.append(c.replace(".",""))
    return [x.replace(".","") for x in equivalentCodes]

negatives = {}
for code in orderedCodes:
    neg = set()
    if cm.is_valid_item(code):
        if len(cm.get_excludes1(code) + cm.get_excludes2(code)) > 0:
            neg.update(get_excludes(code))
        neg.update(get_cousines(code))
    
    toups = []
    for n in neg:
        if n in cod2desc:
            toups.append((n, 0))
    
    random.shuffle(toups)
    #toups = [(cod2desc[x], 0) for x in neg if x in cod2desc] 
    negatives[code] = toups


touples = []

for term, code in pairs:
    upperTerm = term.upper()
    neg = []
    hn = []
    rn = []
    if code in negatives:
        neg = negatives[code]
    if len(neg) > 0:
        neg = sorted(neg, key=lambda x: x[1])
        size = min(5, len(neg))
        i = 0
        while len(hn) < size and i < len(neg):
            if not upperTerm in cod2ents[neg[i][0]]:
                hn.append(neg[i])
                neg[i] = (neg[i][0], neg[i][1] + 1)
            i += 1
    negatives[code] = neg

    nrand = 10 - len(hn)
    while len(rn) < nrand:
        rand = random.choice(tot)
        if not upperTerm in cod2ents[rand]:
            rn.append((rand, 0))
    
    n = hn + rn
    touples.append((
            term, 
            cod2desc[code], 
            cod2desc[n[0][0]], 
            cod2desc[n[1][0]], 
            cod2desc[n[2][0]], 
            cod2desc[n[3][0]], 
            cod2desc[n[4][0]],
            cod2desc[n[5][0]],
            cod2desc[n[6][0]],
            cod2desc[n[7][0]],
            cod2desc[n[8][0]],
            cod2desc[n[9][0]]
        ))

df = pd.DataFrame(touples, columns=["anchor", "positive", "negative_1", "negative_2", "negative_3", "negative_4", "negative_5", "negative_6", "negative_7", "negative_8", "negative_9", "negative_10"])
df.to_csv(OUTPUT_DIR + "definitiveTouples.csv", index=False)

