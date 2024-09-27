#encoder.py

import os
import torch
import torch.nn.functional as F
import numpy as np 
from transformers import GPT2Tokenizer 


"""

    encoder.py 
    1. provide Interface to enter the query for users
    2. use tokenizer to encode query 
    3. save input token ids & input attention mask as header file
    4. you must change the filepath before execution
    
    encoder layer
    
"""

#You can change default file path 
filepath = "./src/"
filename_1 = "gpt2_input.h"

fullpath_1 = filepath + filename_1

#delete file
def delete_file(filepath):
    if os.path.isfile(filepath):
        os.remove(filepath)
        
# Function to create the header file
def create_header_file(filepath, batch, seq_len, gen_len, gen_len_pad, input_ids, attention_mask):
    with open(filepath, 'w') as f:
        f.write('#ifndef GPT2_INPUT_H\n')
        f.write('#define GPT2_INPUT_H\n\n')
        f.write(f'#define IN_BATCH_INIT  {batch}\n')
        f.write(f'#define SEQ_LEN_INIT   {seq_len}\n')
        f.write(f'#define GEN_LEN        {gen_len}\n')
        f.write(f'#define GEN_LEN_PAD    {gen_len_pad}\n\n')
        f.write(f'int input_token_ids[{batch}][{gen_len_pad}] = {{\n')
        
        for batch_index, layer in enumerate(input_ids):
            row_str = ', '.join(map(str, layer.tolist()))
            f.write(f'    {{ {row_str} }}' + (',' if batch_index < batch - 1 else '') + '\n')
        f.write('};\n\n')
        
        f.write(f'int input_attn_mask[{batch}][{gen_len_pad}] = {{\n')
        
        for batch_index, layer in enumerate(attention_mask):
            row_str = ', '.join(map(str, layer.tolist()))
            f.write(f'    {{ {row_str} }}' + (',' if batch_index < batch - 1 else '') + '\n')
        f.write('};\n\n')

        f.write('#endif // GPT2_INPUT_H\n')

# Function for interface
def get_positive_integer(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value >= 0:
                return value
            else:
                print("Please enter a non-negative integer.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

print("\n\n\t\tWELCOME TO THE GPT2-SMALL!!\n\n")

# Get the number of queries
num_queries = get_positive_integer("Enter the number of queries you want to ask: ")

sequences = []

for i in range(1, num_queries + 1):
    query = input(f"\nQuery {i}: ")
    sequences.append(query)

print("\n")
gen_len = get_positive_integer("Enter the length of sequence you want to generate: ")
print("\n\n")

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', clean_up_tokenization_spaces=True)

# Prepare the <pad> token, left-sided padding 
pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the input
encoded_input = tokenizer(sequences, return_tensors='pt', padding=True, truncation=True)
input_ids = encoded_input['input_ids']
attention_mask = encoded_input['attention_mask']

batch = input_ids.size(0)
seq_len = input_ids.size(1)  # Max sequence length in the batch

#padded ids & mask
gen_len_pad = gen_len
remainder = gen_len % 4
if remainder != 0:
	gen_len_pad = gen_len_pad + (4 - remainder)


pad_len = gen_len_pad - seq_len 
padded_input_ids = torch.cat((input_ids, torch.zeros((batch, pad_len), dtype=torch.long)), dim=1)
padded_attention_mask = torch.cat((attention_mask, torch.zeros((batch, pad_len), dtype=torch.long)), dim=1)

delete_file(fullpath_1)

# Create the header file
create_header_file(fullpath_1, batch, seq_len, gen_len, gen_len_pad, padded_input_ids, padded_attention_mask)
