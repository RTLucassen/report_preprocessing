"""
This script downloads the data from Label Studio and prepares the data in the required format for training the model.
"""

from dotenv import load_dotenv
from pathlib import Path
import os
import json
from utils import InputOutputFormat
from utils import LabelStudioClient
from utils import TrainValSplitter
import json
from typing import List
import sys

def download_label_studio_data(dir, label_studio_project_ids, label_studio_api_key, label_studio_url):
    """
    Download the data from Label Studio and save it to a file.

    Args:
    dir (Path): base_dir for the project
    label_studio_project_ids (list): list of project IDs in Label Studio
    label_studio_api_key (str): API key for Label Studio 
    label_studio_url (str): URL for Label Studio (probably a local host URL, e.g., http://localhost:8080/)
    """
    if not label_studio_project_ids or label_studio_api_key is None or label_studio_url is None:
        print("Please ensure all parameters are provided correctly.")
        exit(1)

    client = LabelStudioClient(label_studio_api_key, label_studio_url)

    all_unlabeled_tasks = []
    all_labeled_tasks = []

    for project_id in label_studio_project_ids:
        project = client.get_project(project_id)
        tasks = project.get_tasks()

        for task in tasks:
            if task['annotations']:
                all_labeled_tasks.append(task)
            else:
                all_unlabeled_tasks.append(task)

    # Ensure the directory structure exists
    tasks_dir = Path(dir) / "data" / "label_studio"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    # Save the combined tasks to files
    unlabeled_tasks_file = tasks_dir / "unlabeled_tasks.json"
    with open(unlabeled_tasks_file, "w") as f:
        json.dump(all_unlabeled_tasks, f, indent=4)

    labeled_tasks_file = tasks_dir / "labeled_tasks.json"
    with open(labeled_tasks_file, "w") as f:
        json.dump(all_labeled_tasks, f, indent=4)

    return all_unlabeled_tasks + all_labeled_tasks

def create_input_output_instances(input_file_path: str, labels: List[str], output: bool, context_size: int, 
                                  add_headers: bool, headers: bool, instances: List[int], 
                                  oversampling: bool, class_dict_samples: dict, model_input_file_path: str, 
                                  model_output_file_path: str):
    """
    Function to create the input and output instances for the model.

    Args:
    input_file_path (str): path to the input file
    labels (list): list of labels 
    output (bool): whether to create output instances
    context_size (int): context size for input instances
    add_headers (bool): whether to add headers to the input instances
    headers (list): list of headers to use
    instances (list): list with indices of instances to use
    oversampling (bool): whether to oversample the data
    class_dict_samples (dict): dictionary with class names and oversample ratios or None
    model_input_file_path (str): path to save the input instances
    model_output_file_path (str): path to save the output instances
    """
    class_dict_samples_str = "_".join([f"{key}_{value}" for key, value in class_dict_samples.items()])
    model_input_file_path_alteration = str(model_input_file_path) + f"_context_{context_size}" + f"_headers_{str(add_headers).lower()}_" + str(class_dict_samples_str) + ".json"
    model_output_file_path_alteration = str(model_output_file_path) + f"_context_{context_size}" + f"_headers_{str(add_headers).lower()}_" + str(class_dict_samples_str) + ".json"

    input_output_format = InputOutputFormat(
        json_file_path=str(input_file_path),
        labels=labels,
        output=output,
        context_size=context_size,
        add_headers=add_headers,
        headers=headers,
        instances=instances,
        oversampling=oversampling
    )

    # Load the data
    data = input_output_format.load_data()
    
    # Create the input and output instances
    input, output = input_output_format.create_input_output_instances(data)

    if oversampling:
        if class_dict_samples is not None:
            print("Oversampling the data...")
            input, output = input_output_format.oversample(input, output, class_dict_samples)
        else:
            print("No oversampling dictionary provided. Skipping oversampling...")

    if output:
        for key in input.keys():
            if len(input[key]) != len(output[key]):
                print(f"Mismatch between input and output for key: {key}")
                sys.exit(1)

    # Save the input and output instances to a json file
    # Ensure the directory structure exists
    tasks_dir = Path(model_input_file_path.parent)
    tasks_dir.mkdir(parents=True, exist_ok=True)

    with open(model_input_file_path_alteration, 'w') as f:
        json.dump(input, f, indent=4)
    
    if output:
        with open(model_output_file_path_alteration, 'w') as f:
            json.dump(output, f, indent=4)

def validate_env_vars():
    required_vars = ["LABEL_STUDIO_URL", "LABEL_STUDIO_API_KEY", "PROJECT_IDS"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print("Error: Missing environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nExample how the variables should be defined in the '.env' file:\n")
        print("LABEL_STUDIO_URL=http://localhost:8080/")
        print("LABEL_STUDIO_API_KEY=your_api_key")
        print("PROJECT_IDS=[1,2,3]")
        sys.exit(1)

def prep_data(base_dir: Path, context_size_list: List[int], add_headers_list: List[bool], 
              oversampling_dict: dict, split: str = 'patient', validation_size: float = 0.1):
    """
    Main function to prepare the data for training the model.

    Args:
    base_dir (Path): base directory for the project
    context_size_list (list): list of context sizes to use
    add_headers_list (list): list of boolean values to add headers
    oversampling_dict (dict): dictionary with class names and oversample ratios or None
    split (str): type of split to use (random or patient)
    validation_size (float): size of the validation set
    """
    # Define the paths
    labeled_tasks_file = base_dir  / "data" / "label_studio" / "labeled_tasks.json"   # for caching label studio tasks
    dir_to_patient_id = base_dir / "data" / "train_val_patient_id"                   # for loading parient ids for train-val split

    unlabeled_tasks_file = base_dir / "data" / "label_studio" / "unlabeled_tasks.json"     # load unlabeled tasks from file for creating input-output instances

    train_model_input_file_path = base_dir / "data" / "preprocessed_data" / "training_set_input"    # save train input instances to file
    val_model_input_file_path = base_dir / "data" / "preprocessed_data" / "validation_set_input"    # save val input instances to file
    unlabeled_input_file_path = base_dir / "data" / "preprocessed_data" / "unlabeled_set_input"    # save unlabeled input instances to file

    train_model_output_file_path = base_dir / "data" / "preprocessed_data" / "training_set_output"  # save train output instances to file
    val_model_output_file_path = base_dir / "data" / "preprocessed_data" / "validation_set_output"  # save val output instances to file

    for ADD_HEADERS in add_headers_list:
            for CONTEXT_SIZE in context_size_list:
                # Split the data into training and validation sets
                print("Creating the instance split for the training and validation sets...")
                train_val_splitter = TrainValSplitter()
                train_idx, val_idx = train_val_splitter.create_train_val(labeled_tasks_file=labeled_tasks_file, dir_to_patient_id=dir_to_patient_id,
                                                                        validation_size=validation_size, split=split)

                # Training set
                print("Creating the instances for the training set...")
                create_input_output_instances(input_file_path=labeled_tasks_file, 
                                                labels=labels, 
                                                output=True, 
                                                context_size=CONTEXT_SIZE, 
                                                add_headers=ADD_HEADERS, 
                                                headers=headers, 
                                                instances=train_idx,
                                                oversampling=True,
                                                class_dict_samples=oversampling_dict,
                                                model_input_file_path=train_model_input_file_path, 
                                                model_output_file_path=train_model_output_file_path)

                # Validation set
                print("Creating the instances for the validation set...")
                create_input_output_instances(input_file_path=labeled_tasks_file, 
                                                labels=labels, 
                                                output=True, 
                                                context_size=CONTEXT_SIZE, 
                                                add_headers=ADD_HEADERS, 
                                                headers=headers,
                                                instances=val_idx,
                                                oversampling=False,
                                                class_dict_samples=oversampling_dict,
                                                model_input_file_path=val_model_input_file_path,
                                                model_output_file_path=val_model_output_file_path)

            # Unlabeled set
            print("Creating the instances for the unlabeled set...")
            create_input_output_instances(input_file_path=unlabeled_tasks_file, 
                                            labels=labels, 
                                            output=False, 
                                            context_size=CONTEXT_SIZE, 
                                            add_headers=ADD_HEADERS, 
                                            headers=headers,
                                            instances=None,
                                            oversampling=False,
                                            class_dict_samples=oversampling_dict,
                                            model_input_file_path=unlabeled_input_file_path,
                                            model_output_file_path=None)

                  
if __name__ == "__main__":
    # Env vars for label studio SDK
    BASE_DIR = Path().resolve()
    env_path = BASE_DIR / '.env'
    load_dotenv(dotenv_path=env_path)

    # Load the environment variables
    validate_env_vars()
    LABEL_STUDIO_URL = os.getenv('LABEL_STUDIO_URL')
    LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY')

    labels = [
        "H&E", "IHCplus", "IHC", "MOL", "CON", "ADV", "BRS", "RAD", "CLN", 
        "HIS", "SID", "UNR", "CAL"
    ]

    headers=[
        "structured_report_en", "description_en", 
        "discussion_en", "conclusion_en"
    ]

    context_size_list = [7]
    add_headers_list = [False]
    OVERSAMPLING_DICT= {"HIS": 5, "CAL": 5}
    PROJECT_IDS=eval(os.getenv('PROJECT_IDS'))
    validation_size = 0.1
    
    # split = 'patient'
    # split = 'random'
    split = 'file'

    # Download and save the data to a file
    tasks = download_label_studio_data(BASE_DIR, PROJECT_IDS, LABEL_STUDIO_API_KEY, LABEL_STUDIO_URL)

    # Prepare the data for training the model
    prep_data(BASE_DIR, context_size_list, add_headers_list, OVERSAMPLING_DICT, split, validation_size)