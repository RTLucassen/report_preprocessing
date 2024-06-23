"""
Convert the reports from the original spreadsheet to text files with sentence pairs 
for training the translation model.
"""

import os
import platform
from pathlib import Path

import pandas as pd

# define project root
if platform.system() == 'Linux':
    project_root = Path("")
elif platform.system() == 'Windows':
    project_root = Path(r"")
else:
    NotImplementedError

# define path to report and settings
report_path = project_root/'reports'/'060_reports_discus_split.xlsx'
dataset_path = project_root/'dataset'/'dataset_v8.txt'
source_language = 'nl'
target_language = 'en'
included_categories = ['structured_report', 'description', 'discussion', 'conclusion']
included_iterations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

if __name__ == '__main__':

    # check if reports file exists
    if not os.path.exists(report_path):
        raise FileNotFoundError

    # create output dataset directory if it does not exist yet
    dataset_directory = os.path.split(dataset_path)[0]
    if not os.path.exists(dataset_directory):
        os.makedirs(dataset_directory)
    elif os.path.exists(dataset_path):
        raise FileExistsError

    # load and select reports
    df = pd.read_excel(report_path)
    df = df[df['include_for_translation']]
    df = df[df['iteration'].isin(included_iterations)]

    with open(dataset_path, 'w', encoding='utf-8') as f:
        # loop over reports
        for i, row in df.iterrows():
            for category in included_categories:
                # select the source and target report categories
                source = str(row[f'{category}_{source_language}']).strip()
                target = str(row[f'{category}_{target_language}']).strip()

                # skip empty categories
                if source == 'nan' or target == 'nan':
                    continue
                
                source_sentences = source.replace('. ', '.\n').split('\n')
                target_sentences = target.replace('. ', '.\n').split('\n')

                # skip cases with an unequal number of sentences
                if len(source_sentences) != len(target_sentences):
                    continue
                            
                # loop over sentence pairs
                for source_sentence, target_sentence in zip(source_sentences, target_sentences):
                    if not (len(source_sentence) or len(target_sentence)):
                        continue
                    # include sentence pair
                    f.write(f'{source_sentence}\t{target_sentence}\n')