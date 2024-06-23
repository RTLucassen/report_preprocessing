"""
Download and save original translation language model and tokenizer.
"""

import os
import platform
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# define project root
if platform.system() == 'Linux':
    project_root = Path("")
elif platform.system() == 'Windows':
    project_root = Path("")
else:
    NotImplementedError

# define output path and model checkpoint
output_path = project_root/'models'/'opus-mt-nl-en-original'
model_checkpoint = "Helsinki-NLP/opus-mt-nl-en"

if __name__ == '__main__':

    # create models directory if it does not exist yet
    model_directory = os.path.split(output_path)[0]
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    elif os.path.exists(output_path):
        raise FileExistsError

    # load the model and correponding tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, force_download=True)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, force_download=True) 

    # save the model and corresponding tokenizer
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)