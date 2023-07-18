import os
import requests
import tiktoken
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict


datasets = load_dataset("huggingartists/taylor-swift")

train_percentage = 0.9
validation_percentage = 0.07
test_percentage = 0.03

train, validation, test = np.split(datasets['train']['text'], [int(len(datasets['train']['text'])*train_percentage), int(len(datasets['train']['text'])*(train_percentage + validation_percentage))])

datasets = DatasetDict(
    {
        'train': Dataset.from_dict({'text': list(train)}),
        'validation': Dataset.from_dict({'text': list(validation)}),
        'test': Dataset.from_dict({'text': list(test)})
    }
)

#concatenate all strings in train, so that we can pass in to shakespeare
s = ''
for string in train:
	s += string

train_data = s

m = ''
for string in validation:
	m += string

val_data = m

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
