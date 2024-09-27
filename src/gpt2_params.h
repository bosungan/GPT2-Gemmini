#ifndef GPT2_PARAMS_H
#define GPT2_PARMS_H

#include "gpt2_hparams.h"
#include "../include/gemmini/gemmini_params.h"

//MODEL WEIGHTS
extern const elem_t ln_1_w[LAYER][HLEN];
extern const elem_t ln_1_b[LAYER][HLEN];
extern const elem_t attn_query_w[LAYER][HLEN][HLEN];
extern const elem_t attn_query_b[LAYER][HLEN]; 
extern const elem_t attn_key_w[LAYER][HLEN][HLEN]; 
extern const elem_t attn_key_b[LAYER][HLEN];
extern const elem_t attn_value_w[LAYER][HLEN][HLEN];
extern const elem_t attn_value_b[LAYER][HLEN]; 
extern const elem_t attn_proj_w[LAYER][HLEN][HLEN];
extern const elem_t attn_proj_b[LAYER][HLEN];
extern const elem_t ln_2_w[LAYER][HLEN];
extern const elem_t ln_2_b[LAYER][HLEN];
extern const elem_t mlp_fc_w[LAYER][HLEN][IMLEN];
extern const elem_t mlp_fc_b[LAYER][IMLEN];
extern const elem_t mlp_proj_w[LAYER][IMLEN][HLEN];
extern const elem_t mlp_proj_b[LAYER][HLEN];
extern const elem_t wte[VOCAB][HLEN];
extern       elem_t wte_t[HLEN][VOCAB_PAD];
extern const elem_t wpe[SEQ_MAX][HLEN];
extern const elem_t ln_f_w[HLEN];
extern const elem_t ln_f_b[HLEN];


#endif //GPT2_PARMS_H

