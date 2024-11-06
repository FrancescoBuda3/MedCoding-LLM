import logging
import os
import wandb
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator

from settings import (
    OUTPUT_DIR,
    DATA_DIR
)

from dotenv import load_dotenv
from huggingface_hub import login

if __name__ == "__main__":
    # Set the log level to INFO to get more information
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    # Load variables from the .env file
    load_dotenv()

    HF_TOKEN = os.getenv("HF_TOKEN")
    WB_KEY = os.getenv("WANDB_KEY")
    login(token=HF_TOKEN)
    wandb.login(key=WB_KEY)
    #

    # 1. Load a model to finetune with 2. (Optional) model card data
    # transformer = Transformer("NeuML/pubmedbert-base-embeddings", max_seq_length=64)
    # pooling = Pooling(transformer.get_word_embedding_dimension(), "mean")
    # model = SentenceTransformer(modules=[transformer, pooling])

    model_name = "NeuML/pubmedbert-base-embeddings"
    model = SentenceTransformer(model_name)
    model.max_seq_length = 64

    # model = SentenceTransformer(
    #     "NeuML/pubmedbert-base-embeddings",
    #     model_card_data=SentenceTransformerModelCardData(
    #         language="en",
    #         license="apache-2.0",
    #         model_name="MPNet base trained on AllNLI triplets",
    #     ),
    #     model_kwargs = {"max_seq_length":64},
    # )


    logging.info("Read train dataset: 'ICD-10-CM-HardNegatives'")
    # 2. Load a dataset to finetune on
    dataset = load_dataset("FrancescoBuda/ICD-10-CM-HardNegatives")

    #dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    #test_dataset = dataset["test"]
    train_dataset = dataset["train"]
    logging.info(train_dataset)


    # 3. Define a loss function
    loss = MultipleNegativesRankingLoss(model)

    # # 4. Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.
    # stsb_eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
    # dev_evaluator = EmbeddingSimilarityEvaluator(
    #     sentences1=stsb_eval_dataset["sentence1"],
    #     sentences2=stsb_eval_dataset["sentence2"],
    #     scores=stsb_eval_dataset["score"],
    #     main_similarity=SimilarityFunction.COSINE,
    #     name="sts-dev",
    # )
    # logging.info("Evaluation before training:")
    # dev_evaluator(model)

    # 5. (Optional) Specify training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir="models/pubmedbert-base-embeddings-icd-10-cm-embeddings",
        # Optional training parameters:
        num_train_epochs=3,
        per_device_train_batch_size=128,
        #per_device_eval_batch_size=128,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        # Optional tracking/debugging parameters:
        #eval_strategy="steps",
        #eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        run_name="icd-10-cm-embeddings",  # Will be used in W&B if `wandb` is installed
    )


    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        #eval_dataset=test_dataset,
        loss=loss,
    )
    trainer.train()

    # 8. Save the trained model
    model.save_pretrained("./models/icd-10-cm-embeddings/final")

    # 9. (Optional) Push it to the Hugging Face Hub
    model.push_to_hub("alecocc/icd-10-cm-embeddings")