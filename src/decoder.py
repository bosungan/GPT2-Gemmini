#decoder

import torch # type: ignore
import os
from transformers import GPT2Tokenizer # type: ignore

"""

    decoder.py 
    1. read final token ids from gpt2/output/
    2. decode predicted token sequences by tokenizer 
    3. print out the sequence in user interface 
    
    pre-condition 

    output text file must follow the format

    key [BATCH]
    key [GEN_LEN]
    key [input_token_ids[0][0]]
    key [input_token_ids[0][1]]
    .
    .
    .
    key [input_token_ids[1][GEN_LEN-1]]
"""


# output file path
filepath = './output/generated_token_ids.txt'


# Function to read "token_ids" files in directory and convert to tensor
def read_file_to_tensor(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    batch_size = int(lines[0].strip().split()[1])
    gen_len = int(lines[1].strip().split()[1])
    ids = [int(line.strip().split()[1]) for line in lines[2:]]

    expected_length = batch_size * gen_len
    if len(ids) != expected_length:
        raise ValueError(f"Expected {expected_length} IDs, but found {len(ids)}")


    tensor = torch.tensor(ids, dtype=torch.long).reshape(batch_size, gen_len)

    return tensor
   

# Read text files and convert to tensor
generated_ids = read_file_to_tensor(filepath)

# Load pre-trained tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', clean_up_tokenization_spaces=True)

# Decode the generated text
generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]

# Print generated text
for i, text in enumerate(generated_texts):
    print(f"Generated text {i+1}: {text}\n")
