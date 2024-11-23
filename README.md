# MedCoding-LLM

## Descrizione

questa repository contiene i il codice per replicare gli esperimenti e i dataset realizzati per il progetto di tesi "Automatizzare la Codifica Clinica con i Large Language Models Un Approccio per Migliorare la Codifica ICD in Regimi Low-Resource"

## Utilizzo

Per generare le entità, eseguire lo script `./src/generateEntities.py` fornendo i seguenti due argomenti:

1. **Indice di inizio**: il valore che indica da quale riga iniziare.
2. **Indice di fine**: il valore che indica fino a quale riga generare.

Per generare le coppie, eseguire lo script `./src/generatePairs.py` fornendo i seguenti due argomenti:

1. **Indice di inizio**: il valore che indica da quale riga iniziare.
2. **Indice di fine**: il valore che indica fino a quale riga generare.

Per validare le coppie generate, eseguire lo script `./src/validatePairs.py` fornendo i seguenti due argomenti:

1. **Indice di inizio**: il valore che indica da quale riga iniziare.
2. **Indice di fine**: il valore che indica fino a quale riga generare.

Per eseguire il retriver della fase di pre selezione, utilizzare lo script `./src/preSelection.py`

Per eseguire il la classificazione finale, utilizzare lo script `./src/finalClassification.py`
