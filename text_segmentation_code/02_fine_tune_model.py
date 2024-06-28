"""
This script is used to fine-tune the model on the training data, and save the model.
"""

from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import (
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    IntervalStrategy,
    T5Tokenizer
)
import json
import os
from dotenv import load_dotenv

class EntityRelationDataset(Dataset):
    def __init__(self, encodings, decodings):
        self.encodings = encodings
        self.decodings = decodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.decodings["input_ids"][idx])
        return item

    def __len__(self):
        return len(self.encodings.input_ids)
    

def load_data(input_file, output_file):
    with open(input_file, "r") as f:
        input_instances = json.load(f)
    
    with open(output_file, "r") as f:
        output_instances = json.load(f)
    
    # If you need to preserve the format as a list of lists (for compatibility)
    input_instances_list = list(input_instances.values())
    output_instances_list = list(output_instances.values())

    # Flatten the list of lists
    input = [item for sublist in input_instances_list for item in sublist]
    output = [item for sublist in output_instances_list for item in sublist]
    
    return input, output

def check_mismatches(input_instances, output_instances):
    mismatches = []
    if len(input_instances) != len(output_instances):
        print("Overall length mismatch between input and output.")
    else:
        for index, (input_list, output_list) in enumerate(zip(input_instances, output_instances)):
            if len(input_list) != len(output_list):
                mismatches.append(index)
    
    return mismatches


def train_model(parameter_section, model_name, specific_model_name):
    print(f"Training model for following parameters: {parameter_section}")
    
    # model name based on file name with configurations
    model_save_name = f"{specific_model_name}-{parameter_section}"

    input_file = BASE_DIR / "data" / "preprocessed_data" / str("training_set_input_" + parameter_section + ".json")
    output_file = BASE_DIR / "data" / "preprocessed_data" / str("training_set_input_" + parameter_section + ".json")

    VALIDATION_SET_INPUT_DATA_FILE = BASE_DIR / "data" / "preprocessed_data" / str("validation_set_input_" + parameter_section + ".json")
    VALIDATION_SET_OUTPUT_DATA_FILE = BASE_DIR / "data" / "preprocessed_data" / str("validation_set_output_" + parameter_section + ".json")

    # Step 1 - Load the data
    training_input, training_output = load_data(input_file, output_file)
    validation_input, validation_output = load_data(VALIDATION_SET_INPUT_DATA_FILE, VALIDATION_SET_OUTPUT_DATA_FILE)
    
    # Step 2 - Load the model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    new_tokens = ['<', '<IHCplus>' ,'<CON>', '<H&E>', '<IHC>', '<MOL>', '<ADV>', '<BRS>', '<RAD>', '<CLN>', '<HIS>', '<SID>', '<UNR>', '<CAL>', '<Nline>', '<SENTENCE>']
    tokenizer.add_tokens(new_tokens)

    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Step 3 - Tokenize the data
    train_input_encodings = tokenizer(training_input, truncation=True, padding=True, return_tensors="pt")
    train_output_encodings = tokenizer(training_output, truncation=True, padding=True, return_tensors="pt")
    val_input_encodings = tokenizer(validation_input, truncation=True, padding=True, return_tensors="pt")
    val_output_encodings = tokenizer(validation_output, truncation=True, padding=True, return_tensors="pt")

    # Step 4 - Prepare the dataset
    train_dataset = EntityRelationDataset(train_input_encodings, train_output_encodings)
    val_dataset = EntityRelationDataset(val_input_encodings, val_output_encodings)

    # Step 5 - Prepare the training arguments
    training_args = TrainingArguments(
        output_dir="../results",
        num_train_epochs=10,
        per_device_train_batch_size=12 ,
        logging_dir="../logs",
        evaluation_strategy=IntervalStrategy.EPOCH,  # Evaluate at the end of each epoch
        load_best_model_at_end=True,  # Load the best model in terms of evaluation loss
        metric_for_best_model="eval_loss",  # Use loss for early stopping
        greater_is_better=False,
        save_strategy="epoch",
        save_total_limit=5,
        per_device_eval_batch_size=12,
        learning_rate=1e-4,
    )

    # Step 6 - Prepare the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Step 7 - Train the model
    trainer.train()

    # Step 8 - Save the model
    model.save_pretrained(f"models/{model_save_name}")
    tokenizer.save_pretrained(f"models/{model_save_name}")

# -------------------------------------
# START OF THE MAIN SCRIPT
# -------------------------------------

if __name__ == '__main__':
    BASE_DIR = Path().resolve()

    env_path = BASE_DIR / '.env'
    load_dotenv(dotenv_path=env_path)

    model_name = os.getenv("MODEL_NAME", "google/flan-t5-base")
    specific_model_name = model_name[model_name.find("/") + 1:]

    ### Adjust following variables to specify which model to train
    # Always structured as: context_{context_size}_headers_{headers}_{oversampling}
    # With oversampling being {CLASS}_{OVERSAMPLING_FACTOR} and this joined for all oversampled classes with a "_"
    
    parameter_configurations= [
        "context_7_headers_false_HIS_5_CAL_5"
    ]

    import sys
    f = open("logs.out", 'w')
    sys.stdout = f

    print(f"Training model: {specific_model_name}")
    for parameter in parameter_configurations:
        train_model(parameter, model_name, specific_model_name)
        
    f.close()