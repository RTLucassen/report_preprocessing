"""
This module is used to format the input and output data for the model.
"""

from typing import List, Dict, Tuple
import re
import tqdm
import json
import re
from collections import defaultdict
import random

class InputOutputFormat:
    def __init__(
            self,
            json_file_path: str,
            labels: List[str] = [
                "H&E", "IHCplus", "IHC", "MOL", "CON", "ADV", "BRS", "RAD", "CLN", 
                "HIS", "SID", "UNR", "CAL"
            ],
            output: bool = False,
            context_size: int = 2,
            add_headers: bool = True,
            headers: List[str] = [
                "structured_report_en", "description_en", 
                "discussion_en", "conclusion_en"
            ],
            instances = None,
            oversampling: bool = False
        ):  
        self.json_file_path = json_file_path
        self.labels = labels
        self.begin_labels = [f"<b{label}>" for label in labels]
        self.end_labels = [f"<e{label}>" for label in labels]
        self.output = output
        self.context_size = context_size
        self.add_headers = add_headers
        # reconstruct the normal_class_dict automatically
        self.normal_class_dict = {}
        for begin_label, end_label in zip(self.begin_labels, self.end_labels):
            self.normal_class_dict[begin_label] = ""
            self.normal_class_dict[end_label] = str("<") + end_label[2:-1] + str(">")

        self.headers = headers
        self.instances = instances
        self.oversampling = oversampling

    def __remove_redundant_labels(self, text):
        # This pattern is used to initially detect any label followed by <Nline>
        initial_pattern = "<Nline>|".join([x for x in list(set(self.normal_class_dict.values())) if x != ""]) + "<Nline>"
        
        # Use a flag to check if replacements were made
        replacements_made = True
        while replacements_made:
            replacements_made = False  # Reset the flag for each loop iteration
            labels = re.findall(initial_pattern, text)
            
            for label in set(labels):  # Check each unique label
                # Construct a regex pattern to identify if the label with <Nline> is immediately followed by the same label without encountering another label
                pattern = r"({0})<Nline>([^<]*?{0})".format(label)
                # Perform replacement
                new_text, count = re.subn(pattern, r" \2", text)
                if count > 0:  # If a replacement was made
                    text = new_text
                    replacements_made = True  # Indicate that another pass is needed

        text = re.sub(" :", ":", text)
        text = re.sub(" ,", ",", text)

        return text    
    
    def load_data(self) -> List[Dict]:
        """
        Function to load the data from the json file

        Returns:
        List of dictionaries containing the medical reports and their annotations, where each 
        dictionary contains the following keys: 'id', 'data', 'total_annotations', and 'annotations'
        """
        data = self.__preprocessed_data(self.json_file_path)
        return data
    
    def create_input_output_instances(self, data: List[Dict]) -> Tuple[Dict, Dict]:
        """
        Function to create the input and output instances for the model, the input 
        instances are the sentences that are to be classified and the output instances
        are the labels for the sentences. In "create_input_instances" it is explained
        how the input instances are created and how they are structured and for the
        output instances the same in "create_output_instances". 

        Args:
        data (List[Dict]): List of dictionaries each containing the medical reports 
            and their annotations (if there are any)
        
        Returns:
        Tuple[Dict, Dict]: A tuple containing the input instances and the output instances
        """
    	# Filter the data for e.g. train/validation split
        new_data = []
        if self.instances != None:
            for medical_report in data:
                if medical_report['id'] in self.instances:
                    new_data.append(medical_report)
            data = new_data

        input_instances = {}
        output_instances = {}

        for medical_report in tqdm.tqdm(data):
            input_preprocessing = {}
            # Header pattern
            header_pattern = "|".join([header for header in self.headers])

            # Split the text by the headers and structure the text within a dictionary 
            structured_report, start_end_indices = self.split_text(medical_report['data'], header_pattern)
            for header, (start, end), idx in zip(self.headers, start_end_indices, range(len(self.headers))):
                input_preprocessing[header] = {
                    'text': structured_report[idx],
                    'start': start,
                    'end': end
                }

            for header, text in input_preprocessing.items():
                # The sentence pattern is a positive lookbehind for a period, question mark, or exclamation mark
                # followed by a whitespace and a negative lookahead for a number. The pattern also includes a
                # positive lookbehind for a newline character.
                sentence_pattern = r'(?<=[.?!])(?=\s(?![0-9]))|(?<=\n)'

                # Split the text into sentences
                sentence_splits, start_end_idx_sentences = self.split_text(text['text'], sentence_pattern)
                
                # Filter the list of sentences such that \n and empty strings are removed
                start_end_idx_sentences = [idx for idx, sentence in zip(start_end_idx_sentences, sentence_splits) if sentence.strip() != '']
                sentence_splits = [sentence for sentence in sentence_splits if sentence.strip() != '']

                # Put the splitted text for every header in the dictionary
                input_preprocessing[header]['sentences'] = sentence_splits
                # Put the start and end indices of the sentences for every header in the dictionary
                # but add the start index of the respective header to the start and end indices of the sentences
                input_preprocessing[header]['start_end_idx_sentences'] = [(idx[0] + text['start'], \
                                                                           idx[1] + text['start']) \
                                                                           for idx in start_end_idx_sentences]
                
            all_sentences = []
            for header in self.headers:
                all_sentences.extend(input_preprocessing[header]['sentences'])

            header_index_list = [len(input_preprocessing[header]['sentences']) for header in self.headers]
            header_index_list = [[header]*header_index_list[idx] for idx, header in enumerate(self.headers)]
            header_index_list = [item for sublist in header_index_list for item in sublist]

            input_instances[medical_report['id']] = self.create_input_instances(all_sentences, header_index_list, self.context_size, self.add_headers)
            
            if self.output:
                instances = self.create_output_instances(input_preprocessing, medical_report['annotations'])
                instances = [self.__remove_redundant_labels(instance) for instance in instances]
                output_instances[medical_report['id']] = instances
        return input_instances, output_instances

    def split_text(self, text: str, pattern: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Function to split the text by a pattern and return the split text and the start and 
        end indices of the text

        Args:
        medical_report_string (str): String containing the medical report

        Returns:
        Tuple[List[str], List[Tuple[int, int]]]: A tuple containing the split text and the start and
            end indices of the text
        """        
        split_text = []
        indices = []
        prev_end = 0
        
        for m in re.finditer(pattern, text):
            if prev_end < m.start():
                segment = text[prev_end:m.start()]
                if segment:  # Ensure the segment is not empty
                    split_text.append(segment)
                    indices.append((prev_end, m.start()))
            prev_end = m.end()

        # Capture any remaining text after the last delimiter
        if prev_end < len(text):
            split_text.append(text[prev_end:])
            indices.append((prev_end, len(text)))

        return split_text, indices

    def create_input_instances(self, sentences: list, header_index_list: list, context_size: int = 2, add_headers: bool= False) -> List[str]:
        """
        Function to create the input instances for the model. The input instances are created by adding the 
        context sentences to the sentence that needs to be classified. The context sentences are added in the 
        following way: "Context: {context_before} <SENTENCE> {context_after}". The context_before and context_after
        are the sentences that are present before and after the sentence that needs to be classified. The context_size
        determines how many sentences are used as context.

        Args:
        sentences:         List of strings, where each string contains a sentence
        context_size:      Integer, the number of sentences that are used as context
        header_index_list: List of strings, where each string contains the header of the respective sentence
        add_headers:       Boolean, whether to add the headers in the context or not

        Returns:
        List of strings, where each string contains the input instance for the model
        """
        input_instances = []

        for sentence, idx in zip(sentences, range(len(sentences))):
            # If context size is 0 it means that no context is added and only the sentence is used
            if context_size == 0 and add_headers == False:
                input_instances.append(
                    f"segment_pathology_sentence: '{sentence.strip()}'"
                )
                continue

            # If the index is smaller than the context size, it means that there are not enough sentences before
            # the sentence that needs to be segmented. Therefore we are able to add additional sentences to the
            # context_after
            if idx < context_size:
                
                context_size_after = context_size + (context_size - idx)

                context_before = sentences[:idx]
                context_after = sentences[idx+1:idx+context_size_after+1]

                if add_headers: 
                    headers_intermediate_before = header_index_list[:idx]
                    headers_intermediate_context = header_index_list[idx]
                    headers_intermediate_after = header_index_list[idx+1:idx+context_size_after+1]
                    headers_intermediate = headers_intermediate_before + [headers_intermediate_context] + headers_intermediate_after

            # If the index is larger than the length of the sentences minus the window size, it means that there
            # are not enough sentences after the sentence that needs to be segmented. Therefore we are able to add
            # additional sentences to the context_before
            elif idx > len(sentences) - context_size:
                context_size_before = context_size + (context_size - (len(sentences) - idx))
                starting_index = max(0, idx-context_size_before-1)
                context_before = sentences[starting_index:idx]
                context_after = sentences[idx+1:]

                if add_headers:
                    headers_intermediate_before = header_index_list[starting_index:idx]
                    headers_intermediate_context = header_index_list[idx]
                    headers_intermediate_after = header_index_list[idx+1:]
                    headers_intermediate = headers_intermediate_before + [headers_intermediate_context] + headers_intermediate_after

            # If the index is in the middle of the sentences, we can use the window size as is
            else:
                context_before = sentences[idx-context_size:idx]
                context_after = sentences[idx+1:idx+context_size+1]

                if add_headers:
                    headers_intermediate_before = header_index_list[idx-context_size:idx]
                    headers_intermediate_context = header_index_list[idx]
                    headers_intermediate_after = header_index_list[idx+1:idx+context_size+1]
                    headers_intermediate = headers_intermediate_before + [headers_intermediate_context] + headers_intermediate_after
            
            if add_headers:
                def transform_list(original_list):
                    transformed_list = []
                    seen = set()  # To track seen elements

                    for item in original_list:
                        if item in seen:
                            transformed_list.append('')  # Append empty string for duplicates
                        else:
                            transformed_list.append(item)  # Append the item itself
                            seen.add(item)  # Mark this item as seen

                    return transformed_list
                headers_final = transform_list(headers_intermediate)

                new_context_before = []
                for header, subsentence in zip(headers_final[:len(context_before)], context_before):
                    new_context_before.append(header)
                    new_context_before.append(subsentence)
                
                new_context_before.extend(headers_final[len(context_before):len(context_before)+1])

                new_context_after = []
                for header, subsentence in zip(headers_final[len(context_before)+1:], context_after):
                    new_context_after.append(header)
                    new_context_after.append(subsentence)
                
                context_before = new_context_before
                context_after = new_context_after
           
            context_before = [part.strip() for part in context_before if part.strip() != '']
            context_before = ' '.join(context_before)

            context_after = [part.strip() for part in context_after if part.strip() != '']
            context_after = ' '.join(context_after)          

            input_instances.append(
                f"segment_pathology_sentence: '{sentence.strip()}' \n Context: '{context_before.strip()} <SENTENCE> {context_after.strip()}'"
            )   
        return input_instances
    
    def create_output_instances(self, input_preprocessing: dict, annotations: list) -> list:
        """
        Function to create the output instances for the model by adding labels to the sentence.
        The labels are added based on the annotations that are provided. If there is only 1 annotation
        the labels are added based on that annotation, if there are multiple annotations the labels are
        added based on the last annotation. If there is a ground truth label, the labels are added based
        on the ground truth label.

        Args:
        input_preprocessing (dict): Dictionary containing the preprocessed input, where the keys are the
            headers and the values are dictionaries containing the keys 'text', 'start', 'end', 'sentences',
            and 'start_end_idx_sentences'
        
        annotations (list): List of dictionaries containing the annotations for the total medical report

        Returns:
        List of strings, where each string contains the output instance for the model
        """
        output_instances = []

        # Check if there is only 1 annotation, if so, use that annotation
        if len(annotations) == 1:
            used_annotation = annotations[0]
        else:
            # Check if there is a ground truth label, assuming that only 1 annotation can be the 
            # ground truth
            for annotation in annotations:
                found_ground_truth = False
                if annotation['ground_truth'] == True:
                    used_annotation = annotation
                    found_ground_truth = True
                    break
            if not found_ground_truth:
                # If there is no ground truth label, use the last annotation (most recent annotation)
                used_annotation = annotations[-1]
        
        sorted_annotation = sorted(used_annotation['result'], key=lambda x: x['start'])

        for header, text in input_preprocessing.items():
            if len(text['sentences']) != len(text['start_end_idx_sentences']):
                raise ValueError("The length of the sentences and the start and end indices of the sentences are not equal")
                
            for sentence, start_end_idx in zip(text['sentences'], text['start_end_idx_sentences']):
                output_instances.append(self.label_sentence(sentence, start_end_idx, sorted_annotation))

        return output_instances        

    def label_sentence(self, sentence: str, start_end_idx: Tuple[int, int], annotations: List[Dict]) -> str:
        """
        Function to label the sentences based on the annotations, the parts of the sentences that are
        not annotated are labeled as UNR (Unrelated Information). If the label is "???", it is replaced
        with "QMK" (Question Mark). If the label is "H&E/IHC", it is replaced with "IHCplus" because all 
        code is currently assuming labels of length 3/4. If there are more than one annotation for a
        sentence, each subpart will be divided by <Nline> and labeled separately.

        Example input:
            Skin excision specimen from the back with a largely in situ growing.
        Output:
            Skin excision specimen<CON><Nline>
            from the back<CLN><Nline>
            with a largely in situ growing.<CON>

        Args:
        sentence (str): String containing the sentence
        start_end_idx (Tuple[int, int]): Tuple containing the start and end indices of the sentence
            (relative to the entire medical report)
        annotations (List[Dict]): List of dictionaries containing the annotations for the total medical report

        Returns:
        String containing the labeled sentence
        """
        start_idx, end_idx = start_end_idx

        parts = []
        current_idx = start_idx

        for annotation in annotations:
            if annotation['end'] < start_idx or annotation['start'] > end_idx:
                            continue
            
            # Before the annotation adding
            if current_idx < annotation['start']:
                start_idx_adjusted = current_idx - start_idx
                end_idx_adjusted = annotation['start'] - start_idx
                part_of_sentence = sentence[start_idx_adjusted:end_idx_adjusted].strip()
                if part_of_sentence != '':
                    parts.append(f"<bUNR>{part_of_sentence}<eUNR>")
            
            # The annotation adding
            start_idx_adjusted = max(annotation['start'] - start_idx, 0)
            end_idx_adjusted = min(annotation['end'] - start_idx, end_idx - start_idx)

            part_of_sentence = sentence[start_idx_adjusted:end_idx_adjusted].strip()
            if part_of_sentence != '':
                if annotation['labels'][0] == '???':
                    annotation['labels'][0] = 'QMK'
                    parts.append(f"<b{annotation['labels'][0]}>{part_of_sentence}<e{annotation['labels'][0]}>")
                elif annotation['labels'][0] == "H&E/IHC":
                    annotation['labels'][0] = "IHCplus"
                    parts.append(f"<b{annotation['labels'][0]}>{part_of_sentence}<e{annotation['labels'][0]}>")
                else:
                    parts.append(f"<b{annotation['labels'][0]}>{part_of_sentence}<e{annotation['labels'][0]}>")

            current_idx = annotation['end']

        # After the last annotation adding
        if current_idx < end_idx:
            start_idx_adjusted = current_idx - start_idx
            end_idx_adjusted = end_idx - start_idx
            part_of_sentence = sentence[start_idx_adjusted:end_idx_adjusted].strip()
            if part_of_sentence != '':
                parts.append(f"<bUNR>{part_of_sentence}<eUNR>")

        # Initialize a new list to store the processed parts
        parts_new = []

        for part in parts:
            for label in self.begin_labels:
                # Remove space after a beginning label
                part = re.sub(f"({label})\s+", r"\1", part)  # \1 refers to the captured group (the label itself)
            for label in self.end_labels:
                # Remove space before an ending label
                part = re.sub(f"\s+({label})", r"\1", part)
            parts_new.append(part)
        
        output_instance = "<Nline>".join(parts_new)

        output_instance = self.substitute_labels(output_instance, self.normal_class_dict)

        return output_instance
    
    def substitute_labels(self, instance: str, class_dict: dict) -> Dict:
        """
        Function to substitute the labels in the output instances with a different label
        that is specified in the class_dict

        Args:
        instance (str): String containing the output instance
        class_dict (dict): Dictionary containing the labels to be substituted and their
            corresponding labels

        Returns:
        String containing the output instance with the labels substituted
        """
        for class_name in class_dict.keys():
            instance = re.sub(class_name, class_dict[class_name], instance)
        return instance
    
    def __load_data(self, path: str) -> List[Dict]:
        """
        Load data from a json file into a list of dictionaries
        """
        f = open(path, 'r')

        data = json.load(f)

        f.close()

        return data

    def __filter_data(self, data: List[Dict]) -> List[Dict]:
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

    def __preprocessed_data(self, DATA_DIR: str):
        """
        Load and filter the data
        """
        # Load the data using the DATA_DIR path
        data = self.__load_data(DATA_DIR)

        # Filter the data such that only the necessary information is kept
        data = self.__filter_data(data)

        return data
    
    def oversample(self, input_instances, output_instances, class_dict_samples):
        """
        Function to oversample the data based on the class_dict_samples. The class_dict_samples
        is a dictionary where the keys are the labels and the values are the multiplication factors
        for the respective labels. The multiplication factor determines how many times the label
        should be oversampled.

        Args:
        input_instances (Dict): Dictionary containing the input instances
        output_instances (Dict): Dictionary containing the output instances
        class_dict_samples (Dict): Dictionary containing the labels and their respective multiplication
            factors. 

        Example: 
        class_dict_samples = {
            'H&E': 2,
            'IHC': 3
        }

        Returns:
        Tuple[Dict, Dict]: A tuple containing the oversampled input instances and the oversampled
            output instances
        """
        if class_dict_samples is None:
            return input_instances, output_instances

        # Calculate class balance and track IDs and array indices
        class_balance = defaultdict(list)
        for doc_id, input_list in input_instances.items():
            for idx, input_instance in enumerate(input_list):
                output_instance = output_instances[doc_id][idx]
                pattern = '|'.join([x for x in list(set(self.normal_class_dict.values())) if x != ""])
                pattern = re.sub("\+", r"\+", pattern)

                class_label = re.findall(pattern, output_instance)
                if class_label:
                    class_balance[class_label[0]].append((doc_id, idx))

        # Create copies of the original input and output instances
        oversampled_input = {doc_id: instances.copy() for doc_id, instances in input_instances.items()}
        oversampled_output = {doc_id: instances.copy() for doc_id, instances in output_instances.items()}

        # Oversample the specified classes
        for class_label, multiplication_factor in class_dict_samples.items():
            if class_label in class_balance:
                samples = class_balance[class_label]            
                num_samples = len(samples)
                target_samples = num_samples * multiplication_factor

                # Duplicate the samples
                for _ in range(multiplication_factor - 1):
                    for doc_id, idx in samples:
                        oversampled_input[doc_id].append(input_instances[doc_id][idx])
                        oversampled_output[doc_id].append(output_instances[doc_id][idx])

                # Randomly select additional samples if needed
                if target_samples > num_samples * multiplication_factor:
                    remaining_samples = random.sample(samples, target_samples - num_samples * multiplication_factor)
                    for doc_id, idx in remaining_samples:
                        oversampled_input[doc_id].append(input_instances[doc_id][idx])
                        oversampled_output[doc_id].append(output_instances[doc_id][idx])

        return oversampled_input, oversampled_output