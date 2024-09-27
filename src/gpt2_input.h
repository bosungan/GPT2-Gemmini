#ifndef GPT2_INPUT_H
#define GPT2_INPUT_H

#define IN_BATCH_INIT  2
#define SEQ_LEN_INIT   9
#define GEN_LEN        25
#define GEN_LEN_PAD    28

int input_token_ids[2][28] = {
    { 2061, 318, 262, 3265, 286, 262, 1578, 1829, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 50256, 2061, 318, 262, 3139, 1748, 286, 2869, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
};

int input_attn_mask[2][28] = {
    { 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
};

#endif // GPT2_INPUT_H
