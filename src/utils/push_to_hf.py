from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "../../outputs/checkpoint-3342", # YOUR MODEL USED FOR TRAINING
    max_seq_length = 16000,
    dtype = None,
    load_in_4bit = True,
)

model.push_to_hub_merged("FrancescoBuda/Llama-ICD-coder-1B-merged", tokenizer, save_method = "merged_16bit", token = "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")


