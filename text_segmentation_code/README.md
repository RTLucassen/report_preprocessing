## Configuration

### Environment Setup

To ensure proper functionality, a `.env` file must be present in the root directory. This file includes crucial environment variables needed for the project's operation.

#### How to Create the `.env` File

1. In the project's root directory, create a file named `.env`.
2. Populate the `.env` file with the following essential variables:

```plaintext
# Replace placeholders with your specific settings
PROJECT_IDS=List[int] # Example: [2, 3]
MODEL_NAME="google/flan-t5-base" # Replace with your model's name
LABEL_STUDIO_URL="http://localhost:8080" # Replace with your Label Studio URL
LABEL_STUDIO_API_KEY="YourAPIKeyHere" # Securely add your API key
```

#### Parameters Description

- `PROJECT_IDS`: List of integers, each representing a unique project ID.
- `MODEL_NAME`: Identifier of the model used, such as `"google/flan-t5-base"`.
- `LABEL_STUDIO_URL`: URL of your Label Studio instance.
- `LABEL_STUDIO_API_KEY`: API key for accessing your Label Studio instance. Keep this key secure.

### Note

Ensure all placeholders in the `.env` file are replaced with actual values appropriate for your setup.

## Data Folder Structure

The data directory is organized as follows, with several key folders:

```
\data
|-- label_studio\         # Stores tasks labeled through Label Studio
|   |-- labeled_tasks.json
|   |__ unlabeled_tasks.json
|-- predictions\          # Prediction outputs from models
|   |-- training_predicted_{configurations}.json
|   |-- unlabeled_predicted_{configurations}.json
|   |__ validation_predicted_{configurations}.json
|-- preprocessed_data\    # Input and output data for model training
|   |-- training_set_input_{configurations}.json
|   |-- training_set_output_{configurations}.json
|   |-- unlabeled_set_input_{configurations}.json
|   |__ validation_set_output_{configurations}.json
|__ train_val_patient_id\ # Patient-level data for training/validation
    |-- ids.json
    |-- mapping.json
    |-- reports.json
    |__ train_val_idx.json
```

Initial files (ids.json, mapping.json, and reports.json) that exist in the `train_val_patient_id` folder are used for patient-level mapping. The file `train_val_idx.json` is generated when the script `01_download_prep_data` is run with `split='patient'`. All other files that are shown in the data folder structure are created sequentially by running scripts `01` to `03`.

## Scripts Overview

Each script is designed to perform specific tasks within the project and allows for various customizations:

- **01_download_prep_data**: 
  - **Purpose**: Downloads and preprocesses data from Label Studio.
  - **Adjustable Variables**:
    - `context_size_list`: Specifies the sizes of context (e.g., `list(range(0, 11, 1))`).
    - `add_headers_list`: Indicates whether headers should be added (`[False, True]`).
    - `OVERSAMPLING_DICT`: Defines oversampling rates for classes (e.g., `{"HIS": 5, "CAL": 5}`).
    - `validation_size`: Sets the proportion of data to use for validation (e.g., `0.1`).
    - `split`: Determines the data splitting method (`'patient'`, `'random'`, or `'file'` based on `train_val_idx.json`).

- **02_fine_tune_model**: 
  - **Purpose**: Fine-tunes the specified FLAN-T5 model on the preprocessed data.
  - **Adjustable Variables**:
    - `TrainingArguments`: Allows customization of training parameters such as epochs, batch size, etc.
    - `parameter_configurations`: Configures model training parameters uniquely identified as `context_{context_size}_headers_{headers}_{oversampling}`, where oversampling is detailed as `{CLASS}_{OVERSAMPLING_FACTOR}`.

- **03_create_predictions**: 
  - **Purpose**: Generates predictions on the training, validation, and unlabeled datasets using available models.
  - **Adjustable Variables**: 
    - `models`: Allows the possibility to choose which model should be used to create prediction files for.

- **04_model_evaluation**: 
  - **Purpose**: Evaluates the performance of a specified model.
  - **Adjustable Variables**: 
    - Model name can be specified in the notebook to evaluate different models.

- **05_label_studio_format**: 
  - **Purpose**: Converts the predictions from the FLAN-T5 model into a format compatible with Label Studio.
  - **Adjustable Variables**: 
    - Specifies how predictions are formatted and the details of the import process to Label Studio.

- **eval_of_all_models**:
  - **Purpose**: Evaluates the performance of multiple models.
  - **Adjustable Variables**:
    - You can specify which models should be compared with each other by altering `models`. 