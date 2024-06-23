"""
Convert the translated reports to the label studio format for annotation.
"""

import os
import platform
from pathlib import Path

import pandas as pd
import json
from tqdm import tqdm


# define project root
if platform.system() == 'Linux':
    project_root = Path("")
elif platform.system() == 'Windows':
    project_root = Path("")
else:
    NotImplementedError

# define paths
report_path = project_root/'reports'/'072_reports_discus_split_with_corrections_with_test_set.xlsx'
output_annotation_path = project_root/'annotation'/'reports_10_left_over_cases.json'
output_id_path = project_root/'annotation'/'ids_10_left_over_cases.json'

# define settings
included_columns = ['structured_report_en', 'description_en', 'discussion_en', 'conclusion_en']
included_iterations = [10]
include_for_translation = False

if __name__ == '__main__':

    # create output dataset directory if it does not exist yet
    annotation_directory = os.path.split(output_annotation_path)[0]
    if not os.path.exists(annotation_directory):
        os.makedirs(annotation_directory)
    elif os.path.exists(output_annotation_path):
        raise FileExistsError

    # load reports file and select reports
    df = pd.read_excel(report_path)
    df_selected = df[
        (df['iteration'].isin(included_iterations)) 
        & (df['include_for_translation'] == include_for_translation)
    ]

    i = 1
    ids = {}
    reports = []
    # loop over selected reports and format the information
    for _, row in tqdm(df_selected.iterrows()):
        report = ''
        
        for j, column in enumerate(included_columns):
            report += column+'\n'
            text = str(row[column])
            pad = '\n\n' if j != len(included_columns)-1 else ''
            if text == 'nan':
                text = ''
            report += f'{text}{pad}'

        # store formatted reports
        reports.append({
            "id": i,
            "data": {"text": report},
        })

        # store id to specimen mapping
        ids[i] = f'{row["pa_number"]}_{row["specimen"]}'

        i += 1

    # save reports and mapping
    with open(output_annotation_path, 'w') as f:
        f.write(json.dumps(reports))

    with open(output_id_path, 'w') as f:
        f.write(json.dumps(ids))