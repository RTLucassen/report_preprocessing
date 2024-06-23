"""
Data loading and augmentation.
"""

import random
from pathlib import Path

import platform
import datasets

# define project root
if platform.system() == 'Linux':
    PROJECT_ROOT = Path("")
elif platform.system() == 'Windows':
    PROJECT_ROOT = Path("")
else:
    NotImplementedError


_VERSION = '0.0.1'
_LANGUAGE_PAIRS = [('nl','en')]
_DESCRIPTION = 'Dutch to English sentence pairs from pathology reports.'

# dataset is loaded 5 times
_DATAPATHS = [
    PROJECT_ROOT/'dataset'/'dataset_v8.txt',
    PROJECT_ROOT/'dataset'/'dataset_v8.txt',
    PROJECT_ROOT/'dataset'/'dataset_v8.txt',
    PROJECT_ROOT/'dataset'/'dataset_v8.txt',
    PROJECT_ROOT/'dataset'/'dataset_v8.txt',
    PROJECT_ROOT/'dataset'/'added.txt',
]


def add_letter(string):
    random_letter = list('abcdefghijklmnopqrstuvwxyz')[random.randint(0,25)]
    if len(string):
        characters = list(string)
        characters.insert(random.randint(0, len(characters)), random_letter) 
        corrupted_string = ''.join(characters)
        return corrupted_string
    else:
        return string

def duplicate_character(string):
    if len(string):
        characters = list(string)
        index = random.randint(0, len(characters)-1)
        characters.insert(index+1, characters[index]) 
        corrupted_string = ''.join(characters)
        return corrupted_string
    else:
        return string

def remove_character(string):
    if len(string):
        characters = list(string)
        del characters[random.randint(0, len(characters)-1)]
        corrupted_string = ''.join(characters)
        return corrupted_string
    else:
        return string

def swap_character(string):
    if len(string) > 1:
        characters = list(string)
        index = random.randint(0, len(characters)-2)
        characters[index], characters[index+1] = characters[index+1], characters[index]
        corrupted_string = ''.join(characters)
        return corrupted_string
    else:
        return string

def upper(x, y):
    return x.upper(), y.upper()

def lower(x, y):
    return x.lower(), y.lower()

def title(x, y):
    return x.title(), y.title()

def add_parentheses_around_sentence(x, y):
    return f'({x})', f'({y})'

def add_parentheses_around_word(x, y):
    x = ' '.join([f"({word})" for word in x.split(' ')])
    y = ' '.join([f"({word})" for word in y.split(' ')])
    return x, y

def output_as_input(x, y):
    return y, y


_AUG_CONFIG = {
    'error': {
        'max': 10,
        'prob': 0.2,
        'funcs': [
            add_letter, duplicate_character, remove_character, swap_character,
        ],  
    },
    'format': {
        'max': 2,
        'prob': 0.2,
        'funcs': [
            upper, lower, title, add_parentheses_around_sentence, 
            add_parentheses_around_word, output_as_input,
        ]
    }
}
# 5 * (1 + 19) = 100 epochs
_COPIES = 19


class PathReportConfig(datasets.BuilderConfig):
    def __init__(self, *args, srcl=None, tgtl=None, 
                 aug_config={}, copies=0, **kwargs):
        super().__init__(
            *args,
            name=f"{srcl}-{tgtl}",
            **kwargs,
        )
        self.srcl = srcl
        self.tgtl = tgtl
        self.aug_config = aug_config
        self.copies = copies


class PathReport(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        PathReportConfig(
            srcl=lang1,
            tgtl=lang2,
            aug_config=_AUG_CONFIG,
            copies=_COPIES,
            description=f"Translating {lang1} to {lang2}",
            version=datasets.Version(_VERSION),
        )
        for lang1, lang2 in _LANGUAGE_PAIRS
    ]
    BUILDER_CONFIG_CLASS = PathReportConfig

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "translation": datasets.Translation(languages=(self.config.srcl, self.config.tgtl)),
                },
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"datapaths": _DATAPATHS},
            )
        ]

    def _generate_examples(self, datapaths):
        l1, l2 = self.config.srcl, self.config.tgtl
        for datapath in datapaths:
            with open(datapath, 'r', encoding="utf-8") as f:
                for sentence_counter, pair in enumerate(f):
                    for i in range(self.config.copies+1):
                        x, y = pair.replace('\n', '').split('\t')
                        x = x.strip()
                        y = y.strip()

                        # add spelling errors as data augmentation
                        error_config = self.config.aug_config['error']
                        if len(error_config['funcs']):
                            for _ in range(error_config['max']):
                                if random.random() < error_config['prob']:
                                    func = error_config['funcs'][
                                        random.randint(0, len(error_config['funcs'])-1)
                                    ]
                                    x = func(x)
                        
                        # add format changes as data augmentation
                        format_config = self.config.aug_config['format']
                        if len(format_config['funcs']):
                            for _ in range(format_config['max']):
                                if random.random() < format_config['prob']:
                                    func = format_config['funcs'][
                                        random.randint(0, len(format_config['funcs'])-1)
                                    ]
                                    x, y = func(x, y)

                        result = (
                            (sentence_counter*(self.config.copies+1))+i,
                            {
                                "id": str((sentence_counter*(self.config.copies+1))+i),
                                "translation": {l1: x, l2: y},
                            },
                        )
                        yield result