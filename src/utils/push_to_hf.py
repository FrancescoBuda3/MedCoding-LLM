from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "FrancescoBuda/Llama-ICD-coder-3B", # YOUR MODEL USED FOR TRAINING
    max_seq_length = 16000,
    dtype = None,
    load_in_4bit = True,
)

model.push_to_hub_merged("FrancescoBuda/Llama-ICD-coder-1B-merged-2ep", tokenizer, save_method = "merged_16bit", token = "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")


