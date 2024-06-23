"""
Compare model predictions.
"""

import platform
from pathlib import Path
from typing import Optional

from transformers import (
    pipeline, 
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
)


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

model_1_path = project_root/'models'/'opus-mt-nl-en-finetuned_v11'
model_2_path = project_root/'models'/'opus-mt-nl-en-finetuned_v12_no_aug'
separation_terms = [('', '\n'), ('.', ' ')]

text = 'Het BETREFT een melnaocytairlaesei.'

if __name__ == '__main__':

    # prepare the models
    model_1 = AutoModelForSeq2SeqLM.from_pretrained(model_1_path, local_files_only=True)
    tokenizer_1 = AutoTokenizer.from_pretrained(model_1_path, local_files_only=True)
    pipeline_1 = pipeline('translation', model=model_1, tokenizer=tokenizer_1)

    model_2 = AutoModelForSeq2SeqLM.from_pretrained(model_2_path, local_files_only=True)
    tokenizer_2 = AutoTokenizer.from_pretrained(model_2_path, local_files_only=True)
    pipeline_2 = pipeline('translation', model=model_2, tokenizer=tokenizer_2)

    # separate sentences
    sentences = separate_sentences(text, separation_terms)
    
    # loop over models
    for name, pipe in [(model_1_path.name, pipeline_1), (model_2_path.name, pipeline_2)]:
        # translate each sentence and combine the them
        translation = ''
        for sentence, divide in sentences:
            if sentence == '':
                translated_sentence = sentence
            else:
                translated_sentence = pipe(sentence)[0]['translation_text']
            translation += translated_sentence+divide
        
        # print translation
        print(f'\n{name}:')
        print(translation)
        
