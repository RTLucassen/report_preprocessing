"""
Helper functions to load and filter the data
"""

import json
from typing import Dict, List, Tuple

def load_data(path: str) -> List[Dict]:
    """
    Load data from a json file into a list of dictionaries
    """
    f = open(path, 'r')

    data = json.load(f)

    f.close()

    return data

def filter_data(data: List[Dict]) -> List[Dict]:
    """
    Filter the data such that only the necessary information is kept

    Currently the following components are kept from each sample:
    - id: The id of the sample
    - total_annotations: The total number of annotations for the sample
    - data: The raw text data
    - annotations: A list of annotations, each representing an individual annotation
        Within 'annotations', the following components are kept:
        - id: the id of the annotation
        - ground_truth: a boolean indicating whether the annotation is a ground truth
        - result: the actual annotation

    Format of the returned data:
    [
        {
            'id': str,
            'total_annotations': int,
            'data': str,
            'annotations': [
                {
                    'id': str,
                    'ground_truth': bool,
                    'result': [
                        {
                            start: int,
                            end: int,
                            text: str,
                            labels: [str]
                        }
                    ]
                }
            ]
        }
    ]
    """
    filtered_data = []

    for medical_report in data:

        filtered_medical_report = {
            'id': medical_report['id'],
            'total_annotations': medical_report['total_annotations'],
            'data': medical_report['data']['text'],
            'annotations': []
        }

        for annotation in medical_report['annotations']:
            filtered_annotation = {
                'id': annotation['id'],
                'ground_truth': annotation['ground_truth'],
                'result': [segmentation['value'] for segmentation in annotation['result']]
            }

            filtered_medical_report['annotations'].append(filtered_annotation)

        filtered_data.append(filtered_medical_report)

    return filtered_data

def seperate_data(data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Seperate the data such that text that is annotated three times is considered as a test sample and the remainder as training samples.
    Text that is annotated three times means that we have full agreement on the segmentation of the text.

    Reasoning:
    - The test data is used to evaluate the model
    - The training data is used to train the model
    """
    training_data = []
    test_data = []

    for medical_report in data:
        if medical_report['total_annotations'] == 3:
            test_data.append(medical_report)
        else:
            training_data.append(medical_report)

    return training_data, test_data

def preprocessed_data(DATA_DIR: str):
    """
    Load and filter the data
    """
    # Load the data using the DATA_DIR path
    data = load_data(DATA_DIR)

    # Filter the data such that only the necessary information is kept
    data = filter_data(data)

    return data