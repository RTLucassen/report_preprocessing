"""
Script to convert the predictions to the format required by label studio. Also, 
import the annotations to the existing tasks in label studio.
"""

from typing import List, Dict
import json
from pathlib import Path
import re
from label_studio_sdk import Client
from dotenv import load_dotenv
import os

class PredictionResult:
    def __init__(self, id: int, start: int, end: int, score: float, text: str, labels: List[str], from_name: str = "label", to_name: str = "text", type: str = "labels"):
        self.id = id
        self.start = start
        self.end = end
        self.score = score
        self.text = text
        self.labels = labels
        self.from_name = from_name
        self.to_name = to_name
        self.type = type

    def to_dict(self):
        return {
            "id": self.id,
            "from_name": self.from_name,
            "to_name": self.to_name,
            "type": self.type,
            "value" : {
                "start": self.start,
                "end": self.end,
                "score": self.score,
                "text": self.text,
                "labels": self.labels
            }
        }

class Prediction:
    def __init__(self, model_version: str, score: float, results: List[PredictionResult]):
        self.model_version = model_version
        self.score = score
        self.results = results

    def to_dict(self):
        return {
            "model_version": self.model_version,
            "score": self.score,
            "result": [result.to_dict() for result in self.results]
        }

class DataEntry:
    def __init__(self, text: str, predictions: List[Prediction], id: int):
        self.task = id
        self.data = {"text": text}
        self.predictions = predictions

    def to_dict(self):
        return {
            "task": self.task,
            "data": self.data,
            "predictions": [prediction.to_dict() for prediction in self.predictions]
        }

def object_to_json(obj): 
    return json.dumps(obj, indent=3)

def parse_labels(labeled_sentence):
    """
    Parse the labels from the labeled sentence and maintain order information.
    """
    parts = labeled_sentence.split("<Nline>")
    labeled_texts = []  # Use a list to maintain order
    for part in parts:
        match = re.match(r"(.+)<(.{3,4})>", part)
        if match:
            text = match.group(1).strip()
            label = match.group(2)
            labeled_texts.append({'text': text, 'label': label})
    return labeled_texts

def reformat_predictions_for_import_existing_task(predictions: List[Dict]):
    """
    If a task already exists in label studio, we can import the annotations to the existing task.

    Current format of "predictions"":
    [
        {
        "task": task_id,
        "data": {"text": text},
        "predictions": [{
            "model_version": model_version,
            "score": score,
            "result": [annotation_results]
            }]
        }
    ]

    The following format is required:
    [
        {
        "task": task_id,
        "result": [annotation_results]
        "model_version": model_version
        }
    ]
    """
    new_predictions = []

    for prediction in predictions:
        task_id = prediction["task"]
        result = prediction["predictions"][0]["result"]
        model_version = prediction["predictions"][0]["model_version"]

        new_predictions.append({
            "task": task_id,
            "result": result,
            "model_version": model_version
        })
    
    return new_predictions

def import_annotations_existing_tasks(predictions: List[Dict], label_studio_project_ids: List[int], label_studio_api: str, label_studio_url: str):
    """
    Import the annotations to existing tasks in label studio.
    https://github.com/HumanSignal/label-studio-sdk/blob/master/examples/import_preannotations/import_preannotations.ipynb

    Args:
    - predictions (List[Dict]): List of predictions in the format:
    [
        {
        "task": task_id,
        "data": {"text": text},
        "predictions": [{
            "model_version": model_version,
            "score": score,
            "result": [annotation_results]
            }]
        }
    ]
    - label_studio_project_id List(int): List of project IDs in label studio
    - label_studio_api (str): The API key for label studio
    - label_studio_url (str): The URL of the label studio instance

    Returns:
    The response from the label studio API
    """ 
    ls_client = Client(label_studio_url, label_studio_api)

    # Reformat the predictions
    new_predictions = reformat_predictions_for_import_existing_task(predictions)

    for label_studio_project_id in label_studio_project_ids:
        project = ls_client.get_project(int(label_studio_project_id))
        task_ids = project.get_tasks_ids()
        prediction_for_this_project = [prediction for prediction in new_predictions if prediction["task"] in task_ids]
        if len(prediction_for_this_project) == 0:
            continue
        else:
            print(f"Importing annotations to project {label_studio_project_id}")
            print(project.create_predictions(prediction_for_this_project))
    
    return "Annotations imported successfully."



if __name__ == "__main__":

    BASE_DIR = Path().resolve()

    # Define the data
    ORIGINAL_EXPORT = (
        BASE_DIR / "data" / "label_studio" / "unlabeled_tasks.json"
    )

    OUTPUT_DATA_FILE = (
        BASE_DIR / "data" / "predictions" / "unlabeled_predicted_context_7_headers_false_HIS_5_CAL_5.json"
    )

    original_export = json.loads(ORIGINAL_EXPORT.read_text())
    output_data_file = json.loads(OUTPUT_DATA_FILE.read_text())
    preprocess_config = str(OUTPUT_DATA_FILE)[str(OUTPUT_DATA_FILE).index("unlabeled_predicted")+len("unlabeled_predicted")+1:]

    # Remove annotations from the original export
    entries = []
    global_id = 0
    error_count = 0

    # Adjusting the main loop to process ordered predictions
    for task in original_export:
        idx = task['id']
        original_text = task["data"]["text"]
        sentence_predictions = output_data_file.get(str(idx))
        if sentence_predictions is None:
            continue
        
        prediction_results = []
        current_position = 0
        for prediction in sentence_predictions:
            labeled_texts = parse_labels(prediction)
            for item in labeled_texts:
                text = item['text']
                label = item['label']
                # Manual adjustment
                if label == "IHCplus":
                    label = "H&E/IHC"    
    
                start = original_text.find(text, current_position)
                if start == -1:
                    print(f"idx: {idx}")
                    print(f"Could not find the text: '{text}' in the sentence: '{original_text}'")
                    error_count += 1
                    continue
                end = start + len(text)
                current_position = end  # Update to search for the next text beyond this one
                
                prediction_result = PredictionResult(
                    id=str(global_id),
                    start=start,
                    end=end,
                    score=0.70,
                    text=text,
                    labels=[label]
                )
                global_id += 1
                prediction_results.append(prediction_result)
        
        predictions = [Prediction(model_version=preprocess_config, score=0.5, results=prediction_results)]
        entry = DataEntry(text=original_text, predictions=predictions, id=idx)
        entries.append(entry)

    # Save the entries to a file
    output_file = BASE_DIR / "data" / "predictions.json"
    output_file.write_text(object_to_json([entry.to_dict() for entry in entries]))
    print(f"Predictions saved to {output_file}")

    if error_count > 0:
        print(f"Encountered {error_count} errors while processing the data.")

    with open(output_file, "r") as f:
        predictions = json.load(f)

    # Import the annotations to label studio
    env_path = BASE_DIR / ".env"
    load_dotenv(dotenv_path=env_path)
    
    LABEL_STUDIO_PROJECT_ID = eval(os.getenv("PROJECT_IDS"))
    LABEL_STUDIO_API = os.getenv("LABEL_STUDIO_API_KEY")
    LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")   

    import_annotations_existing_tasks(predictions, LABEL_STUDIO_PROJECT_ID, LABEL_STUDIO_API, LABEL_STUDIO_URL)