"""
This module is used to validate the model's predictions. It is used to compare the actual labels with the predicted labels.
"""

import json
import re
from sklearn.metrics import confusion_matrix, classification_report 
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from .create_predictions import ModelPredictor

class TextSegmentationValidator:
    def __init__(self, labels, model_name, base_dir, model_dir, data_preprocessed_dir, data_predictions_dir):
        self.model_name = model_name
        self.preprocess_configurations = str("_") + model_name[model_name.find("context"):]

        self.data_preprocessed_dir = data_preprocessed_dir

        # Ensure paths are constructed correctly using Path
        base_dir = Path(base_dir)
        self.training_y_path = base_dir / data_preprocessed_dir / f"training_set_output{self.preprocess_configurations}.json"
        self.training_pred_path = base_dir / data_predictions_dir / f"training_predicted{self.preprocess_configurations}.json"
        self.validation_y_path = base_dir / data_preprocessed_dir / f"validation_set_output{self.preprocess_configurations}.json"
        self.validation_pred_path = base_dir / data_predictions_dir / f"validation_predicted{self.preprocess_configurations}.json"

        self.model_dir = Path(model_dir)  
        
        self.labels = [f"<{label}>" for label in labels]

        self.check_requirements()

        self.evaluation_results = {}
        
    def load_data(self, path_y, path_pred):
        with open(path_y, "r") as f:
            y_true = json.load(f)
        
        with open(path_pred, "r") as f:
            y_pred = json.load(f)
        
        return y_true, y_pred
        
    def check_requirements(self):
        """
        Every model should have the following files:

        Training set:
        - ../data/preprocessed_data/training_set_output_{preprocess_configuration}.json
        - ../data/predictions/training_predicted_{preprocess_configuration}.json
        
        Validation set:
        - ../data/preprocessed_data/validation_set_output_{preprocess_configurations}.json
        - ../data/predictions/validation_predicted_{preprocess_configuration}.json

        If the latter of the two files is not present, we will create it. 
        """    
        missing_files = False
        print(f"Checking requirements for {self.model_name}:")
        
        # Check for output files existence
        for path in [self.training_y_path, self.validation_y_path]:
            if not path.exists():
                print(f"- Missing required output file: {path}")
                missing_files = True

        # Check for prediction files existence
        for path in [self.training_pred_path, self.validation_pred_path]:
            if not path.exists():
                print(f"- Missing prediction file: {path}")
                print("  Creating prediction file...")
                try:
                    dataset_type = "training" if "training" in str(path) else "validation"
                    predictor = ModelPredictor(self.model_name, dataset_type, str(self.data_preprocessed_dir), str(self.model_dir))
                    predictor.predict()
                    print("  Prediction file created successfully.")
                except Exception as e:
                    print(f"  Failed to create prediction file: {e}")
                    missing_files = True
        
        if missing_files:
            print("\nSome requirements are missing. Please check the logs above.")
        else:
            print("All requirements met successfully.\n")


        # Proceed to load data only if all files are present
        if not missing_files:
            self.y_true_train, self.y_pred_train = self.load_data(self.training_y_path, self.training_pred_path)
            self.y_true_val, self.y_pred_val = self.load_data(self.validation_y_path, self.validation_pred_path)
            print("Data loaded successfully.")

    def parse_text(self, input_text):
        """
        This parser should translate the input text into a format such that for every word in the input text,
        the parser should be able to identify the class label it belongs to. Since we only care about the words, it will not take into account the 
        interpunctuations and other symbols.

        The input text is expected to be in the following format:
        "Text belonging to class 1<class1><Nline>class 2 text<class2><Nline>...<classN>"

        The output of the parser are two lists:
        1. A list of lists containing the words in the input text
        2. A list containing the class labels for each word in the input text

        Example:
        input_text = "This is a test<H&E><Nline>Another test<IHC><Nline>Final test<MOL>"
        output = (["This", "is", "a", "test", "Another", "test", "Final", "test"], 
                    ["<H&E>", "<H&E>", "<H&E>", "<H&E>", "<IHC>", "<IHC>", "<MOL>", "<MOL>"]) 
        """
        text = []
        labels_per_word = []

        # Split the input text by newline
        split_by_newline = input_text.split("<Nline>")

        # Initialize a pattern to find class labels 
        pattern = re.compile(f"({'|'.join(self.labels)})")

        for sentence in split_by_newline:
            split_by_label = re.split(pattern, sentence)
            if len(split_by_label) != 3:
                raise ValueError(f"This is unexpected. Check the following sentence: {sentence}")
            if split_by_label[2] != "":
                raise ValueError(f"This is unexpected. Check the following sentence: {sentence}")
            
            sentence = split_by_label[0]
            label = split_by_label[1]

            # Split the sentence by space
            split_by_space = sentence.split()

            for word in split_by_space:
                text.append(word)    
                labels_per_word.append(label)

        return text, labels_per_word

    def validate(self, dataset_name):
        """
        Validate the model's predictions for a given dataset (training or validation).
        
        Parameters:
        - dataset_name: A string identifier for the dataset ('training' or 'validation').
        - y_true: The true labels.
        - y_pred: The predicted labels.
        """
        y_true = self.y_true_train if dataset_name == "training" else self.y_true_val
        y_pred = self.y_pred_train if dataset_name == "training" else self.y_pred_val

        format_errors = []
        actual_labels = []
        predicted_labels = []

        self.actual_labels_per_medical_report = {}
        self.predicted_labels_per_medical_report = {}

        for medical_report in y_true.keys():
            if medical_report not in y_pred.keys():
                raise ValueError(f"The medical report {medical_report} is not found in the predictions.")
            
            if len(y_true[medical_report]) != len(y_pred[medical_report]):
                raise ValueError(f"The number of sentences in the actual text and the predicted text do not match for the medical report {medical_report}.")
            
            self.actual_labels_per_medical_report[medical_report] = []
            self.predicted_labels_per_medical_report[medical_report] = []
            
            for sentence_y, sentence_pred, idx in zip(y_true[medical_report], y_pred[medical_report], range(len(y_true[medical_report]))):
                try:
                    actual_text, actual_labels_per_sentence = self.parse_text(sentence_y)
                    predicted_text, predicted_labels_per_sentence = self.parse_text(sentence_pred)
                except:
                    format_errors.append((sentence_y, sentence_pred, medical_report, idx))
                    continue
                
                if len(actual_text) != len(predicted_text):
                    format_errors.append((actual_text, predicted_text, medical_report, idx))
                    continue
                
                actual_labels.extend(actual_labels_per_sentence)
                predicted_labels.extend(predicted_labels_per_sentence)

                self.actual_labels_per_medical_report[medical_report].extend(actual_labels_per_sentence)
                self.predicted_labels_per_medical_report[medical_report].extend(predicted_labels_per_sentence)
        
        self.evaluation_results[dataset_name] = {
            "total_sentences": len([item for sublist in y_true.values() for item in sublist]),
            "format_errors": format_errors,
            "conf_matrix": confusion_matrix(actual_labels, predicted_labels, labels=self.labels),
            "classification_report": classification_report(actual_labels, predicted_labels, labels=self.labels, zero_division=0, output_dict=True),
            "accuracy": np.mean(np.array(actual_labels) == np.array(predicted_labels))
        }

    def plot_confusion_matrix(self, dataset_name, normalize=False):
        """
        Plots the confusion matrix for the specified dataset.
        
        Parameters:
        - dataset_name: The name of the dataset ('training' or 'validation') for which to plot the confusion matrix.
        - normalize: If True, normalize the confusion matrix.
        """
        if dataset_name not in self.evaluation_results:
            print(f"No evaluation results found for dataset: {dataset_name}")
            return

        cm = self.evaluation_results[dataset_name]['conf_matrix']

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".3f"
        else:
            fmt = "d"  # Integer formatting

        sns.set_theme(font_scale=0.8)
        plt.figure(figsize=(9, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            cbar=True,
            xticklabels=self.labels,
            yticklabels=self.labels
        )
        plt.title(f"Confusion Matrix - {dataset_name.title()}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()

    def explain_errors(self, dataset_name):

        if dataset_name not in self.evaluation_results:
            print(f"No evaluation results found for dataset: {dataset_name}")

        format_errors = self.evaluation_results[dataset_name]['format_errors']

        if len(format_errors) == 0:
            print(f"No format errors found for dataset: {dataset_name}")
            return
        
        print(f"Format errors found for dataset: {dataset_name}")
        print(f"A total of {len(format_errors)} errors are found from the total of {self.evaluation_results[dataset_name]['total_sentences']} sentences.")
        for error in format_errors:
            print(f"Actual text: {error[0]}")
            print(f"Predicted text: {error[1]}")
            print(f"Medical report: {error[2]}")
            print(f"Sentence index: {error[3]}")
            print("\n")