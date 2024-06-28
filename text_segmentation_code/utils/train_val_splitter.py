"""
This module performs the splitting of the training and validation set, on a patient level.
"""

import pandas as pd
import re
from typing import Dict, List
from pathlib import Path
import json 
import random 
from .input_output_formatter import InputOutputFormat
import os


class TrainValSplitter:
    def __init__(self):
        self.input_instances = None
        self.output_instances = None
        self.input_output_format = None
        self.mapping_label_studio_pa_number = {}

    def __n_amount_of_classes(self, output_instances: Dict[int, List[str]], class_names) -> pd.DataFrame:
        """
        Function to count the amount of classes in the output instances

        Args:
        output_instances:  Dictionary containing the output instances
        class_names: List containing the class names

        Returns:

        """
        dataframe_dict = {}

        class_names = [name for name in list(set(class_names)) if name != '']

        for key, sentences in output_instances.items():
            dataframe_dict[key] = {}
            for class_name in class_names:
                dataframe_dict[key][class_name] = 0
            for sentence in sentences:
                for class_name in class_names:
                    dataframe_dict[key][class_name] += len(re.findall(class_name, sentence))
        
        return pd.DataFrame(dataframe_dict).T
    
    def __create_header_dict(self, input_instances: Dict[int, List[str]], headers: list) -> pd.DataFrame:
        """
        Function to create a dictionary that for each medical report 
        contains whether a certain header is present in the input instances

        Args:
        input_instances:  Dictionary containing the input instances
        headers: List containing the headers

        Returns:
        DataFrame containing the header dictionary
        """
        header_dict = {}
        for key, instances in input_instances.items():
            
            header_dict[key] = {}
            for header in headers:
                header_dict[key][header] = 0

            for instance in instances:
                for header in headers:
                    if re.search(header, instance):
                        header_dict[key][header] = 1
        return pd.DataFrame(header_dict).T
    
    def __merge_dicts(self, dict1, dict2):
        """
        Merge two dictionaries based on the values of the first dictionary
        and the keys of the second dictionary.

        Example: 
        dict1 = {"a": "b"}
        dict2 = {"b": "c"} 
        merge_dicts(dict1, dict2) -> {"a": "c"}
        """
        result = {}
        for key, value in dict1.items():
            if value in dict2:
                result[key] = dict2[value]
        return result
    
    def __flip_dict(self, dictionary):
        """
        Flip a dictionary such that the keys become the values and the values become the keys
        and taking into account that the values are not unique

        Example:
        dictionary = {"a": "b", "c": "b"}

        flip_dict(dictionary) -> {"b": ["a", "c"]}
        """
        result = {}
        for key, value in dictionary.items():
            if value in result:
                result[value].append(key)
            else:
                result[value] = [key]
        return result
    

    def __train_validation_split(self, input_instances, output_instances, report_id_to_patient_mapping, patient_to_report_id_mapping, headers,
                                 validation_size: float = 0.1, structured_report_en_ratio: float = 0.1, discussion_en_ratio: float = 0.1, 
                                 mol_ratio: float = 0.1, random_state = 42):
        """
        Function to split the input and output instances into a training and validation set
        while taking into account a patient level split and the structure of the medical reports.

        Args:
            input_instances (dict): Dictionary containing the input instances
            output_instances (dict): Dictionary containing the output instances
            report_id_to_patient_mapping (dict): Dictionary containing the mapping from report id to patient
            patient_to_report_id_mapping (dict): Dictionary containing the mapping from patient to report id
            headers (list): List of headers to check in reports
            validation_size (float): The ratio of validation samples
            structured_report_en_ratio (float): The ratio of structured report samples
            discussion_en_ratio (float): The ratio of description samples
            mol_ratio (float): The ratio of minority class samples
            random_state (int): The random state for the random sampling

        Returns:
            tuple: train_instances, validation_instances
        """
        random.seed(random_state)
        validation_n = int(validation_size * len(input_instances))
        validation_instances = []

        # Create a overview of medical reports and the presence of content within the headers    
        header_dataframe = self.__create_header_dict(input_instances, headers)

        # Step 1: Structured Report Inclusion
        structured_report_ids = header_dataframe[header_dataframe['structured_report_en'] > 0].index
        structured_report_en_n = int(structured_report_en_ratio * validation_n)
        val_structured_report_ids = random.sample(list(structured_report_ids), structured_report_en_n)

        # Step 2: Discussion Inclusion
        discussion_ids = header_dataframe[header_dataframe['discussion_en'] > 0].index
        discussion_en_n = int(discussion_en_ratio * validation_n)
        val_discussion_ids = random.sample(list(discussion_ids), discussion_en_n)

        # This loop is to include a whole patient in the validation set, 
        # it omits that a patient has a report in the training set and another in the validation set
        for id in list(set(val_structured_report_ids + val_discussion_ids)):
            # Check if the patient has more than one report
            patient = report_id_to_patient_mapping[id]

            ids_per_patient = patient_to_report_id_mapping[patient]

            for id in ids_per_patient:
                validation_instances.append(int(id))

        # Step 3: MOL Inclusion
        n_classes_dataframe = self.__n_amount_of_classes(output_instances, self.input_output_format.normal_class_dict.values())
        mol_overview = n_classes_dataframe[n_classes_dataframe['<MOL>'] > 0]
        mol_ids = list(mol_overview.index)
        
        mol_count = 0
        mol_n = int(mol_ratio * validation_n)
        iter_count = 0

        while mol_count < mol_n and iter_count < 1000:
            id = random.choice(mol_ids)
            
            # This loop is to include a whole patient in the validation set, 
            # it omits that a patient has a report in the training set and another in the validation set
            if id not in validation_instances:
                mol_count += mol_overview.loc[id, '<MOL>']
                
                patient = report_id_to_patient_mapping[id]

                ids_per_patient = patient_to_report_id_mapping[patient]

                for id in ids_per_patient:
                    validation_instances.append(int(id))
            iter_count += 1
        
        iter_count = 0
        
        # Step 4: Random Sampling
        while len(validation_instances) < validation_n and iter_count < 1000:
            id = random.choice(list(input_instances.keys()))

            # This loop is to include a whole patient in the validation set,
            # it omits that a patient has a report in the training set and another in the validation set
            if id not in validation_instances:

                patient = report_id_to_patient_mapping[id]

                ids_per_patient = patient_to_report_id_mapping[patient]

                for id in ids_per_patient:
                    validation_instances.append(int(id))
            
            iter_count += 1
        
        train_instances = list(set(input_instances.keys()) - set(validation_instances))

        return sorted(train_instances), sorted(validation_instances)

    def __match_original_report_ls_tasks(self, original_reports, original_ids, ls_tasks):

        def __substitute_value_from_dict(ids, reports):
            for id, report in zip(ids.values(), reports):
                report['id'] = id
            return reports

        for (id_file, id_content), (report_file, report_content) in zip(original_ids.items(), original_reports.items()):
            original_reports[report_file] = __substitute_value_from_dict(id_content, report_content)

        report_id_mapping = {}
        for task in ls_tasks:
            report_id_mapping[task['id']] = 0
            label_studio_report_text = task['data']
            found_report = False

            for report_file, report_list in original_reports.items():
                for idx in range(len(report_list)):
                    report_text = report_list[idx]['data']['text']
                    if report_text == label_studio_report_text:
                        report_id_mapping[task['id']] = report_list[idx]['id']
                        task['original_id'] = report_list[idx]['id']
                        self.mapping_label_studio_pa_number[task['id']] = report_list[idx]['id']
                        # Efficiently remove the matched report from the list
                        original_reports[report_file].pop(idx)
                        found_report = True
                        break
                if found_report:
                    break

            if found_report:
                continue
            
            print("Could not find the report: ", task['id'])
        return ls_tasks, report_id_mapping

    def create_train_val(self, labeled_tasks_file: Path, dir_to_patient_id: Path,
                         validation_size: float = 0.1, split: str = 'patient', 
                         structured_report_en_ratio: float = 0, discussion_en_ratio: float = 0, 
                         mol_ratio: int = 0, random_state: int = 42):
        """
        Function to create the training and validation split based on the labeled tasks file
        and the patient id mapping. The split can be done on a patient level or randomly.

        Args:
        labeled_tasks_file: Path to the labeled tasks file
        dir_to_patient_id: Path to the directory containing the patient id mapping
        validation_size: The ratio of validation samples
        split: The type of split (patient, random or file)
        structured_report_en_ratio: The ratio of structured report samples
        discussion_en_ratio: The ratio of discussion samples
        mol_ratio: The ratio of minority class samples
        random_state: The random state for the random sampling

        Returns:
        Tuple containing the training and validation indices
        """

        if split == 'file':
            with open(dir_to_patient_id / 'train_val_idx.json', 'r') as f:
                train_val_idx = json.load(f)
                return train_val_idx['train_idx'], train_val_idx['val_idx']
            
        # Step 1: Define the configuration
        labels=[
            "H&E", "IHCplus", "IHC", "MOL", "CON", "ADV", "BRS", "RAD", "CLN", 
            "HIS", "SID", "UNR", "CAL"
        ]
        context_size=2
        add_headers=True
        headers=[
            "structured_report_en", "description_en", 
            "discussion_en", "conclusion_en"
        ]
        
        # Step 2: Preprocess the labeled data
        self.input_output_format = InputOutputFormat(
            json_file_path=labeled_tasks_file,
            labels=labels,
            output=True,
            context_size=context_size,
            add_headers=add_headers,
            headers=headers
        )

        data = self.input_output_format.load_data()

        self.input_instances, self.output_instances = self.input_output_format.create_input_output_instances(data)
        
        if split == 'patient':
            print("Matching the labeled reports from label studio to the original reports for a patient level split...")
            # Load the mapping from report id to patient id
            id_data = {}
            report_data = {}

            # Load JSON files directly
            for file_name in os.listdir(dir_to_patient_id):
                if file_name.startswith("ids_"):
                    with open(dir_to_patient_id / file_name, 'r') as file:
                        id_data[file_name] = json.load(file)
                elif file_name.startswith("reports_"):
                    with open(dir_to_patient_id / file_name, 'r') as file:
                        report_data[file_name] = json.load(file)

            data, label_studio_to_original_mapping = self.__match_original_report_ls_tasks(report_data, id_data, data)

            with open(dir_to_patient_id / 'mapping.json', 'r') as file:
                patient_level_mapping = json.load(file)

            patient_level_mapping = self.__merge_dicts(label_studio_to_original_mapping, patient_level_mapping)
            
            patient_to_report_id_mapping = self.__flip_dict(patient_level_mapping)

            train_idx, val_idx = self.__train_validation_split(input_instances=self.input_instances, \
                                                            output_instances=self.output_instances,
                                                            report_id_to_patient_mapping=patient_level_mapping,
                                                            patient_to_report_id_mapping=patient_to_report_id_mapping, 
                                                            validation_size=validation_size, 
                                                            structured_report_en_ratio=structured_report_en_ratio, 
                                                            discussion_en_ratio=discussion_en_ratio, 
                                                            mol_ratio=mol_ratio, 
                                                            random_state=random_state, 
                                                            headers=headers)
            
            print("Amount of training instances: ", len(train_idx), "\nAmount of validation instances: ", len(val_idx))

            with open(dir_to_patient_id / "train_val_idx.json", "w") as f:
                json.dump({"train_idx": train_idx, "val_idx": val_idx}, f, indent=4)
            
            return train_idx, val_idx

        elif split == 'random':
            print("Random split...")
            # Random split
            idx = list(self.input_instances.keys())
            # Set the random state and shuffle the indices
            random.seed(random_state)
            random.shuffle(idx)
            # Split the indices
            split_idx = int(len(idx) * (1 - validation_size))
            # Get the training and validation indices
            train_idx = idx[:split_idx]
            val_idx = idx[split_idx:]
            # Save the indices
            print("Amount of training instances: ", len(train_idx), "\nAmount of validation instances: ", len(val_idx))
            return train_idx, val_idx
        
        else:
            raise ValueError("Split type not recognized. Please choose between 'patient', 'random' or 'file'.")
    
    def eval_train_val(self, train_idx: List[int], val_idx: List[int], verbose: bool = False):
        """
        Function to evaluate the split of the training and validation set. Since we have a specific structure
        in the medical reports, we are interested in (1) the different structures that are present in the training
        and validation set. Moreover, we are interested in (2) the distribution of the classes in the training and
        validation set.

        Args:
        train_idx: List containing the training instances
        val_idx: List containing the validation instances
        verbose: Boolean indicating whether to print the results

        Returns:
        Tuple containing the header and class distribution analysis (both pandas dataframe)
        """
        headers=[
            "structured_report_en", "description_en", 
            "discussion_en", "conclusion_en"
        ]
        # Load train instances
        train_input_instances = {key: self.input_instances[key] for key in train_idx}
        train_output_instances = {key: self.output_instances[key] for key in train_idx}

        # Load validation instances
        val_input_instances = {key: self.input_instances[key] for key in val_idx}
        val_output_instances = {key: self.output_instances[key] for key in val_idx}

        # Structure analysis
        train_header_df = self.__create_header_dict(train_input_instances, headers)
        train_header_df = train_header_df.groupby(headers).size().reset_index(name='n_reports_train').set_index(headers)

        val_header_df = self.__create_header_dict(val_input_instances, headers)
        val_header_df = val_header_df.groupby(headers).size().reset_index(name='n_reports_val').set_index(headers)

        header_df = pd.concat([train_header_df, val_header_df], axis=1).fillna(0)
        header_df = header_df.astype(int)
        header_df['normalized_n_train'] = header_df['n_reports_train'] / header_df['n_reports_train'].sum()
        header_df['normalized_n_val'] = header_df['n_reports_val'] / header_df['n_reports_val'].sum()  

        if verbose:
            print("Structure Analysis")
            print(header_df)

        # Class distribution analysis
        train_class_df = self.__n_amount_of_classes(train_output_instances, self.input_output_format.normal_class_dict.values())
        description_train = train_class_df.describe()
        sum_row_train = pd.DataFrame(train_class_df.sum()).transpose()
        sum_row_train.index = ['sum']
        description_train = pd.concat([description_train, sum_row_train])
        description_train = description_train.T[['count', 'mean', 'sum']].rename(columns={
            'count': 'n_report_train',
            'mean': 'mean_sentences_train',
            'sum': 'total_sentences_train'
        })

        val_class_df = self.__n_amount_of_classes(val_output_instances, self.input_output_format.normal_class_dict.values())
        description_val = val_class_df.describe()
        sum_row_val = pd.DataFrame(val_class_df.sum()).transpose()
        sum_row_val.index = ['sum']
        description_val = pd.concat([description_val, sum_row_val])
        description_val = description_val.T[['count', 'mean', 'sum']].rename(columns={
            'count': 'n_report_val',
            'mean': 'mean_sentences_val',
            'sum': 'total_sentences_val'
        })

        class_df = pd.concat([description_train, description_val], axis=1).fillna(0)[['n_report_train', 'n_report_val', 'total_sentences_train', 'total_sentences_val', 'mean_sentences_train', 'mean_sentences_val']]

        if verbose:
            print("Class Distribution Analysis")
            print(class_df)

        return header_df, class_df        

