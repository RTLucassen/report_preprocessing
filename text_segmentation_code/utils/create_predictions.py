"""
This module is used to predict on the data using a trained model.
"""

from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import json
import re
import time
import tqdm

class ModelPredictor:
    def __init__(self, model_name, file_to_predict_on, data_dir = "./data/preprocessed_data/", model_dir="./models/"):
        """
        Initializes the predictor with the model and the file for prediction.

        Parameters:
        - model_name: Name of the model directory.
        - file_to_predict_on: either training, validation or unlabeled data.
        """
        self.base_dir = Path().resolve()
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.preprocess_configurations = str("_") + model_name[model_name.find("context"):]

        self.path_model_file = self.base_dir / model_dir / self.model_name
        self.path_prediction_file = self.base_dir / data_dir / str(file_to_predict_on + "_set_input" + self.preprocess_configurations + ".json")

        # Load the model and tokenizer from the specified directory
        self.model = T5ForConditionalGeneration.from_pretrained(str(self.path_model_file))
        self.tokenizer = T5Tokenizer.from_pretrained(str(self.path_model_file))
        
        # Set device to CUDA if available, else CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def load_data(self, input_file):
        """
        Loads the input data from a specified file.

        Parameters:
        - input_file: Path to the JSON file containing the data.

        Returns:
        A dictionary of the input instances.
        """
        with open(input_file, "r") as f:
            input_instances = json.load(f)
        return input_instances

    def predict_sentence_batch(self, input_text):
        """
        Predicts the output for a batch of sentences using the model.

        Parameters:
        - input_text: A list of sentences for prediction.

        Returns:
        A list of predicted sentences.
        """
        # Tokenize input text; pad & truncate to handle variable lengths
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # Generate predictions with the model
        output_sequences = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=512,
            early_stopping=False,
            # num_beams=6
        )

        # Decode the predictions into readable text
        return self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

    def snug_class_labels(self, text):
        # Pattern to match spaces right before '<' or right after '>' and remove them
        pattern = r'\s+(?=<)|(?<=>)\s+'
        return re.sub(pattern, '', text)

    def predict(self, batch_size=10):
        """
        Predicts the text for the entire dataset specified at initialization, in batches.

        Parameters:
        - batch_size: The number of sentences to predict in one go.

        Returns:
        A dictionary of predicted texts for each report.
        """
        input_data = self.load_data(self.path_prediction_file)
        predicted_text = {}

        start = time.time()
        for id, report in tqdm.tqdm(input_data.items()):
            predicted_report_texts = []
            current_batch = []
            for sentence in report:
                current_batch.append(sentence)
                if len(current_batch) == batch_size:
                    # Predict when the batch reaches the specified batch size
                    batch_predictions = self.predict_sentence_batch(current_batch)
                    predicted_report_texts.extend(batch_predictions)
                    current_batch = []  # Reset the batch
            if current_batch:
                # Ensure any remaining sentences are processed
                batch_predictions = self.predict_sentence_batch(current_batch)
                predicted_report_texts.extend(batch_predictions)

            # Clean up class labels in the predicted text
            predicted_report_texts = [self.snug_class_labels(sentence) for sentence in predicted_report_texts]
            predicted_text[id] = predicted_report_texts

        print(f"Time taken: {time.time() - start} seconds")

        # Ensure the directory structure exists
        predictions_dir = self.base_dir / self.data_dir.parent / "predictions" 
        predictions_dir.mkdir(parents=True, exist_ok=True)

        # Save the predictions to a file
        if "training" in str(self.path_prediction_file):
            output_path = self.base_dir / self.data_dir.parent / "predictions" / f"training_predicted{self.preprocess_configurations}.json"
        if "validation" in str(self.path_prediction_file):
            output_path = self.base_dir / self.data_dir.parent / "predictions" / f"validation_predicted{self.preprocess_configurations}.json"
        if "unlabeled" in str(self.path_prediction_file):
            output_path = self.base_dir / self.data_dir.parent / "predictions" / f"unlabeled_predicted{self.preprocess_configurations}.json"


        with open(output_path, "w") as f:
            json.dump(predicted_text, f, indent=4)
        return predicted_text
    