"""
Finetune the translation model starting from the downloaded model weigths.
"""

import os
import platform
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
)


def preprocessing_function(examples):
    inputs = [example['nl'] for example in examples["translation"]]
    targets = [example['en'] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=512, truncation=True)
    return model_inputs


# define project root
if platform.system() == 'Linux':
    project_root = Path("")
    repo_root = Path("")
elif platform.system() == 'Windows':
    project_root = Path("")
    repo_root = Path(r"")
else:
    NotImplementedError

# define paths
pretrained_model_path = project_root/'models'/'opus-mt-nl-en-original'
output_model_path = project_root/'models'/'opus-mt-nl-en-finetuned_v11'
dataset_code_path = repo_root/'translation_code'/'PathReport.py'

if __name__ == '__main__':

    # check if the output file already exists
    if os.path.exists(output_model_path):
        raise FileExistsError

    # load pretrained models
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, local_files_only=True) 

    # load dataset, tokenize it and prepare collator
    dataset = load_dataset(str(dataset_code_path), trust_remote_code=True)
    tokenized_dataset = dataset.map(preprocessing_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_model_path,
        learning_rate=1e-4,
        per_device_train_batch_size=32,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1, # actually 100, see PathReport.py
    )

    # define trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # train model based on training arguments and save it afterwards
    trainer.train()
    trainer.save_model(output_model_path)