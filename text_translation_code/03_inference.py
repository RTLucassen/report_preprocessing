"""
Perform inference using the trained language model.
"""

import os
import platform
import random
from pathlib import Path
from typing import Optional

import pandas as pd
from transformers import (
    pipeline, 
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
)
from tqdm import tqdm


def separate_sentences(text: str, separation_terms: list[tuple[str, str]]
) -> list[tuple[str, Optional[str]]]:
    """
    Separate text into sentences based on the separation terms.

    Args:
        text:  text to be separated.
        separation terms:  list with tuples, where each tuple contains a part 
            that should be added to the sentence and a part to divide the sentence. 
            Together, the parts form one separation term.

    Returns:
        sentences:  list of tuples with sentences and dividing parts.
    """
    sentences = [(text.strip(), '')]
    # loop over separation terms
    for keep, divide in separation_terms:
        term = keep+divide
        new_sentences = []
        # loop over sentences
        for sentence, end in sentences:
            # split sentence based on separation term
            sub_sentences = sentence.split(term)
            for i, sub_sentence in enumerate(sub_sentences):
                # store separated sentences
                if i == len(sub_sentences)-1:
                    new_sentences.append((sub_sentence, end))
                else:
                    new_sentences.append((sub_sentence+keep, divide))
        # update the sentences
        sentences = new_sentences
    
    return sentences


# define project root
if platform.system() == 'Linux':
    project_root = Path("")
elif platform.system() == 'Windows':
    project_root = Path("")
else:
    NotImplementedError

# define paths
report_path = project_root/'reports'/'060_reports_discus_split.xlsx'
output_report_path = project_root/'reports'/'061_reports_discus_split.xlsx'
model_path = project_root/'models'/'opus-mt-nl-en-finetuned_v11'

# define settings
N_cases_to_translate = 10000
source_language = 'nl'
target_language = 'en'
included_categories = ['structured_report', 'description', 'discussion', 'conclusion']
separation_terms = [('', '\n'), ('.', ' ')]
selected_specimens = False
selected_years = [2020, 2019]


if __name__ == '__main__':

    # check if the output file already exists
    if os.path.exists(output_report_path):
        raise FileExistsError

    # set seed
    random.seed(0)

    # prepare the model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    pipe = pipeline('translation', model=model, tokenizer=tokenizer)

    # load reports and select the dutch, untranslated ones
    df = pd.read_excel(report_path)
    iteration = int(max(df['iteration'].fillna(0).tolist()))+1

    # get indices
    df_selection = df[(df['original_in_dutch'] == True) & (df['include_for_translation'] == False)]
    if selected_specimens:
        df_selection = df_selection[df_selection['selected'] == True]
    if selected_years is not None:
        if len(selected_years):
            df_selection = df_selection[df_selection['year'].isin(selected_years)]
    indices = list(df_selection.index)
    random.shuffle(indices)

    # loop over cases and translate the original reports
    for i in tqdm(indices[:min(N_cases_to_translate, len(indices))]):
        df.at[i, 'iteration'] = iteration
        for category in included_categories:
            original = str(df[f'{category}_{source_language}'].iloc[i])
            if original != 'nan':
                # separate sentences
                sentences = separate_sentences(original, separation_terms)
                # translate each sentence and combine the them
                translation = ''
                for sentence, divide in sentences:
                    if sentence == '':
                        translated_sentence = sentence
                    else:
                        translated_sentence = pipe(sentence)[0]['translation_text']
                    translation += translated_sentence+divide
                df.at[i, f'{category}_{target_language}'] = translation

    # save translation results
    df.to_excel(output_report_path, index=False)