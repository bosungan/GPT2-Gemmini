#ifndef GPT2_H
#define GPT2_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "gpt2_params.h"
#include "gpt2_hparams.h"
#include "gpt2_input.h"
#include "riscv_math.h"

#include "../include/gemmini/gemmini.h"
#include "../include/gemmini/gemmini_nn.h"

#define ROUND_UP_TO_MUL_OF_4(x) (((x) + 3) & ~3)
#define GET_DIM_I(x) 		(tiled_matmul_type == CPU ? ROUND_UP_TO_MUL_OF_4(x) : x)
#define printf_v(fmt, ...)  do { if (verbose) printf(fmt, ##__VA_ARGS__); } while (0)

#define TRANSPOSE_MATRIX(src, dst, rows, cols)     \
    do {                                           \
        for (size_t i = 0; i < rows; ++i) {        \
            for (size_t j = 0; j < cols; ++j) {    \
                dst[j][i] = src[i][j];             \
            }                                      \
        }                                          \
    } while (0)


//defined in .c file
extern enum tiled_matmul_type_t tiled_matmul_type;
extern bool check;
extern bool verbose;

//MODEL LAYER - THESE VARIABLES SUPPORT MODEL OPERATIONS
elem_t input_pos_ids[IN_BATCH_INIT][GEN_LEN_PAD];
elem_t block_input_init[IN_BATCH_INIT][GEN_LEN_PAD][HLEN];
elem_t block_input[IN_BATCH_INIT][GEN_LEN_PAD][HLEN];
elem_t input_norm[IN_BATCH_INIT][GEN_LEN_PAD][HLEN] row_align(1);
elem_t query_layer[IN_BATCH_INIT][GEN_LEN_PAD][HLEN] row_align(1);
elem_t key_layer[IN_BATCH_INIT][GEN_LEN_PAD][HLEN] row_align(1);
elem_t value_layer[IN_BATCH_INIT][GEN_LEN_PAD][HLEN] row_align(1);
elem_t query_newshape[IN_BATCH_INIT][GEN_LEN_PAD][NUM_HEAD][SIZ_HEAD];
elem_t key_newshape[IN_BATCH_INIT][GEN_LEN_PAD][NUM_HEAD][SIZ_HEAD];
elem_t value_newshape[IN_BATCH_INIT][GEN_LEN_PAD][NUM_HEAD][SIZ_HEAD];
elem_t query_perm[IN_BATCH_INIT][NUM_HEAD][GEN_LEN_PAD][SIZ_HEAD] row_align(1);
elem_t key_perm[IN_BATCH_INIT][NUM_HEAD][GEN_LEN_PAD][SIZ_HEAD];
elem_t key_perm_t[IN_BATCH_INIT][NUM_HEAD][SIZ_HEAD][GEN_LEN_PAD] row_align(1);
elem_t value_perm[IN_BATCH_INIT][NUM_HEAD][GEN_LEN_PAD][SIZ_HEAD] row_align(1);
elem_t attention_score[IN_BATCH_INIT][NUM_HEAD][GEN_LEN_PAD][GEN_LEN_PAD];
elem_t mask_base[GEN_LEN_PAD][GEN_LEN_PAD];
elem_t mask_used[IN_BATCH_INIT][GEN_LEN_PAD][GEN_LEN_PAD];
elem_t attention_probs[IN_BATCH_INIT][NUM_HEAD][GEN_LEN_PAD][GEN_LEN_PAD];
elem_t context_layer[IN_BATCH_INIT][NUM_HEAD][GEN_LEN_PAD][SIZ_HEAD];
elem_t context_layer_perm[IN_BATCH_INIT][GEN_LEN_PAD][NUM_HEAD][SIZ_HEAD];
elem_t context_layer_output[IN_BATCH_INIT][GEN_LEN_PAD][HLEN] row_align(1);
elem_t context_layer_proj[IN_BATCH_INIT][GEN_LEN_PAD][HLEN] row_align(1);
elem_t context_resadd[IN_BATCH_INIT][GEN_LEN_PAD][HLEN] row_align(1);
elem_t input_mlp[IN_BATCH_INIT][GEN_LEN_PAD][HLEN] row_align(1);
elem_t mlp_fc_output[IN_BATCH_INIT][GEN_LEN_PAD][IMLEN] row_align(1);
elem_t mlp_gelu[IN_BATCH_INIT][GEN_LEN_PAD][IMLEN] row_align(1);
elem_t mlp_proj[IN_BATCH_INIT][GEN_LEN_PAD][HLEN] row_align(1);
elem_t block_output[IN_BATCH_INIT][GEN_LEN_PAD][HLEN] row_align(1);
elem_t block_output_last[IN_BATCH_INIT][GEN_LEN_PAD][HLEN];
elem_t output_norm[IN_BATCH_INIT][GEN_LEN_PAD][HLEN] row_align(1);
elem_t last_output_norm[IN_BATCH_INIT][4][HLEN]; // used for acceleration of model_logits
elem_t last_output_logits[IN_BATCH_INIT][4][VOCAB_PAD]; //used for acceleration of model logits 
elem_t output_logits[IN_BATCH_INIT][GEN_LEN_PAD][VOCAB_PAD] row_align(1);
elem_t output_probs[IN_BATCH_INIT][GEN_LEN_PAD][VOCAB];
int    output_token_ids[IN_BATCH_INIT][GEN_LEN_PAD];


//KV cache 
elem_t KV_Cache[LAYER][KV_NUM][IN_BATCH_INIT][GEN_LEN_PAD][HLEN];
elem_t last_input_norm[IN_BATCH_INIT][4][HLEN];
elem_t last_key_layer[IN_BATCH_INIT][4][HLEN];
elem_t last_value_layer[IN_BATCH_INIT][4][HLEN];

//PERFORMANCE ANALYSIS 
uint64_t start, end;
uint64_t attn_norm_cycles = 0, attn_QKV_cycles = 0, attn_reshape_cycles = 0, attn_score_cycles = 0, attn_masking_cycles = 0;
uint64_t attn_softmax_cycles = 0, attn_ctx_cycles = 0, attn_merge_cycles = 0, attn_proj_cycles = 0, attn_resadd_cycles = 0; 
uint64_t mlp_norm_cycles = 0, mlp_fc_cycles = 0, mlp_gelu_cycles = 0, mlp_proj_cycles = 0, mlp_resadd_cycles = 0; 
uint64_t model_norm_cycles = 0, model_logits_cycles = 0, model_softmax_cycles = 0, model_argmax_cycles = 0;  

void print_cycles_info(){
    uint64_t total_cycles = attn_norm_cycles + attn_QKV_cycles + attn_reshape_cycles + attn_score_cycles + attn_masking_cycles +
                            attn_masking_cycles + attn_softmax_cycles + attn_ctx_cycles + attn_merge_cycles + attn_proj_cycles + attn_resadd_cycles + 
                            mlp_norm_cycles + mlp_fc_cycles + mlp_gelu_cycles + mlp_proj_cycles + mlp_resadd_cycles +
                            model_norm_cycles + model_logits_cycles + model_softmax_cycles + model_argmax_cycles;

    printf("\n\n<CYCLE INFO>\n\n");
    printf("Total                cycles: %llu\n", total_cycles);
    printf("attention norm       cycles: %llu (%d%%)\n", attn_norm_cycles, (attn_norm_cycles * 100) / total_cycles);
    printf("attention QKV        cycles: %llu (%d%%)\n", attn_QKV_cycles, (attn_QKV_cycles * 100) / total_cycles);
    printf("attention reshape    cycles: %llu (%d%%)\n", attn_reshape_cycles, (attn_reshape_cycles * 100) / total_cycles);
    printf("attention score      cycles: %llu (%d%%)\n", attn_score_cycles, (attn_score_cycles * 100) / total_cycles);
    printf("attention masking    cycles: %llu (%d%%)\n", attn_masking_cycles, (attn_masking_cycles * 100) / total_cycles);
    printf("attention softmax    cycles: %llu (%d%%)\n", attn_softmax_cycles, (attn_softmax_cycles * 100) / total_cycles);
    printf("attention context    cycles: %llu (%d%%)\n", attn_ctx_cycles, (attn_ctx_cycles * 100) / total_cycles);
    printf("attention merge      cycles: %llu (%d%%)\n", attn_merge_cycles, (attn_merge_cycles * 100) / total_cycles);
    printf("attention projection cycles: %llu (%d%%)\n", attn_proj_cycles, (attn_proj_cycles * 100) / total_cycles);
    printf("attention resadd     cycles: %llu (%d%%)\n", attn_resadd_cycles, (attn_resadd_cycles * 100) / total_cycles);
    printf("mlp       norm       cycles: %llu (%d%%)\n", mlp_norm_cycles, (mlp_norm_cycles * 100) / total_cycles);
    printf("mlp       fc         cycles: %llu (%d%%)\n", mlp_fc_cycles, (mlp_fc_cycles * 100) / total_cycles);
    printf("mlp       gelu       cycles: %llu (%d%%)\n", mlp_gelu_cycles, (mlp_gelu_cycles * 100) / total_cycles);
    printf("mlp       projection cycles: %llu (%d%%)\n", mlp_proj_cycles, (mlp_proj_cycles * 100) / total_cycles);
    printf("mlp       resadd     cycles: %llu (%d%%)\n", mlp_resadd_cycles, (mlp_resadd_cycles * 100) / total_cycles);
    printf("model     norm       cycles: %llu (%d%%)\n", model_norm_cycles, (model_norm_cycles * 100) / total_cycles);
    printf("model     logits     cycles: %llu (%d%%)\n", model_logits_cycles, (model_logits_cycles * 100) / total_cycles);
    printf("model     softmax    cycles: %llu (%d%%)\n", model_softmax_cycles, (model_softmax_cycles * 100) / total_cycles);
    printf("model     argmax     cycles: %llu (%d%%)\n", model_argmax_cycles, (model_argmax_cycles * 100) / total_cycles);
    printf("\n\n");
}


/*
	MODEL LAYER FUNCTION
*/

void allocate_model_layer_input(int IN_BATCH, int SEQ_LEN, int flag)
{
    int start_idx = (flag == USE_NONE) ? 0 : SEQ_LEN - 1;

    //create pos_ids based on token_ids
    for (int i = 0; i < IN_BATCH; i++) {
        int pos = 0;
        for (int j = start_idx; j < SEQ_LEN; j++) {
            if (input_token_ids[i][j] == PAD_ID)   input_pos_ids[i][j] = 0;
            else if(flag == USE_NONE)              input_pos_ids[i][j] = pos++;
            else                                   input_pos_ids[i][j] = input_pos_ids[i][j - 1] + 1;
        }
    }


    //create block_input_init
	if(flag == USE_NONE){
		for (int b = 0; b < IN_BATCH; b++)
		{
			for (int gen = 0; gen < GEN_LEN_PAD; gen++)
			{
				for(int idx = 0; idx < HLEN; idx++)
				{
					block_input_init[b][gen][idx] = 0; 
				}
			}
		}
	}


    for (int b = 0; b < IN_BATCH; b++)
    {
        for (int seq = start_idx; seq < SEQ_LEN; seq++)
        {
            int pos_id   = input_pos_ids[b][seq];
            int token_id = input_token_ids[b][seq];

            for (int idx = 0; idx < HLEN; idx++)
            {
                block_input_init[b][seq][idx] = wte[token_id][idx] + wpe[pos_id][idx];
            }
        }
    }
}

/*
    BLOCK FORWARD FUNCTIONS
    1.  Normalization in attention layer
    2.  QKV matrix multiplication
    3.  QKV matrix reshaping into multi-head 
    4.  Getting attention score
    5.  Masking 
    6.  Softmax
    7.  Getting context layer by matmul with value matrix
    8.  Merging multi-head
    9.  Projection in attention layer
    10. Residual addition in attention layer
    11. Normalization in MLP layer
    12. Forwarding Fully Connected layer
    13. Activation layer(GELU)
    14. Projection in MLP layer
    15. Final residual addition in MLP layer
*/

void attn_norm(int layer, int IN_BATCH, int SEQ_LEN, float eps);
void attn_QKV(int layer, int IN_BATCH, int SEQ_LEN);
void attn_reshape(int IN_BATCH, int SEQ_LEN);	
void attn_score(int IN_BATCH, int SEQ_LEN);
void attn_masking(int IN_BATCH, int SEQ_LEN);
void attn_softmax(int IN_BATCH, int SEQ_LEN);
void attn_ctx(int IN_BATCH, int SEQ_LEN);
void attn_ctx_reshape(int IN_BATCH, int SEQ_LEN);
void attn_proj(int layer, int IN_BATCH, int SEQ_LEN);
void attn_resadd(int IN_BATCH, int SEQ_LEN);
void mlp_norm(int layer, int IN_BATCH, int SEQ_LEN, float eps);
void mlp_fc(int layer, int IN_BATCH, int SEQ_LEN);
float gelu(float x);
void mlp_gelu_forward(int IN_BATCH, int SEQ_LEN);
void mlp_proj_forward(int layer, int IN_BATCH, int SEQ_LEN);
void mlp_resadd(int IN_BATCH, int SEQ_LEN);

//normalization 1
void attn_norm(int layer, int IN_BATCH, int SEQ_LEN, float eps)
{  
   start = read_cycles();
   for (int b = 0; b < IN_BATCH; b++)
	{
		for (int seq = 0; seq < SEQ_LEN; seq++)
		{
			// calculate each mean, var based on each vector(HLEN)
			float avg = 0.0f;
			float var = 0.0f;
			float sum = 0.0f;
			float sqsum = 0.0f;

			for (int idx = 0; idx < HLEN; idx++)
			{
				sum   += block_input[b][seq][idx];
				sqsum += block_input[b][seq][idx] * block_input[b][seq][idx];
			}

			avg = sum / HLEN;
			var = sqsum / HLEN - avg * avg;

            //normalize
			for (int idx = 0; idx < HLEN; idx++)
			{
				input_norm[b][seq][idx] = (block_input[b][seq][idx] - avg) / sqrtf(var + eps) * ln_1_w[layer][idx] + ln_1_b[layer][idx];
			}
		}
	}

    end = read_cycles();
    attn_norm_cycles += end - start;
    printf_v("ATTN - NORM FINISHED\n");
}

//get Q, K, V matrix by matmul
void attn_QKV(int layer, int IN_BATCH, int SEQ_LEN)
{
    start = read_cycles();

    //MATMUL to get QKV matrix
    for (int b = 0; b < IN_BATCH; b++)
	{
		for (int i = 0; i < SEQ_LEN; i++)
		{
			for (int j = 0; j < HLEN; j++)
			{
				query_layer[b][i][j] = attn_query_b[layer][j];
                key_layer[b][i][j]   = attn_key_b[layer][j];
				value_layer[b][i][j] = attn_value_b[layer][j];

				for (int k = 0; k < HLEN; k++)
				{
					query_layer[b][i][j] += input_norm[b][i][k] * attn_query_w[layer][k][j];
                    key_layer[b][i][j]   += input_norm[b][i][k] * attn_key_w[layer][k][j];
                    value_layer[b][i][j] += input_norm[b][i][k] * attn_value_w[layer][k][j];
				}
			}
		}
    }

    end = read_cycles();
    attn_QKV_cycles += end - start;
    printf_v("ATTN - QKV FINISHED\n");
}

//reshaping
//step 1: split into multi-head
//step 2: permutation --> exchange 2nd, 3rd dimension (head, seq_num)
void attn_reshape(int IN_BATCH, int SEQ_LEN)
{   
    start = read_cycles();

    //split into multi-head
    for (int b = 0; b < IN_BATCH; b++)
	{
		for (int seq = 0; seq < SEQ_LEN; seq++)
		{
			for (int num = 0; num < NUM_HEAD; num++)
			{
				for (int siz = 0; siz < SIZ_HEAD; siz++)
				{

					query_newshape[b][seq][num][siz] = query_layer[b][seq][siz + num * SIZ_HEAD];
					key_newshape[b][seq][num][siz]   = key_layer[b][seq][siz + num * SIZ_HEAD];
					value_newshape[b][seq][num][siz] = value_layer[b][seq][siz + num * SIZ_HEAD];
				}
			}
		}
	}

    //permutation
    //key_perm_t is transpose of key_perm
    for (int b = 0; b < IN_BATCH; b++)
	{
		for (int seq = 0; seq < SEQ_LEN; seq++)
		{
			for (int num = 0; num < NUM_HEAD; num++)
			{
				for (int siz = 0; siz < SIZ_HEAD; siz++)
				{
					query_perm[b][num][seq][siz] = query_newshape[b][seq][num][siz];
					key_perm[b][num][seq][siz]   = key_newshape[b][seq][num][siz];
					key_perm_t[b][num][siz][seq] = key_perm[b][num][seq][siz];
					value_perm[b][num][seq][siz] = value_newshape[b][seq][num][siz];
				}
			}
		}
	}

    end = read_cycles();
    attn_reshape_cycles += end - start;
    printf_v("ATTN - RESHAPE FINISHED\n");
}


//get normalized attention score
//(normalized attention score) = Q * K_T / sqrt(d_k)
void attn_score(int IN_BATCH, int SEQ_LEN)
{   
    start = read_cycles();
    
	for (int b = 0; b < IN_BATCH; b++)
	{
		for (int num = 0; num < NUM_HEAD; num++)
		{
			for (int orow = 0; orow < SEQ_LEN; orow++)
			{
				for (int ocol = 0; ocol < SEQ_LEN; ocol++)
				{
                    float temp = 0.0f;

					for (int siz = 0; siz < SIZ_HEAD; siz++)
					{
                        temp += query_perm[b][num][orow][siz] * key_perm[b][num][ocol][siz];
					}

                    temp = temp / sqrtf(SIZ_HEAD);
                    attention_score[b][num][orow][ocol] = temp;
				}
			}
        }
    }

    end = read_cycles();
    attn_score_cycles += end - start;
    printf_v("ATTN - SCORE FINISHED\n");
}

//masking
void attn_masking(int IN_BATCH, int SEQ_LEN)
{   
    start = read_cycles();

	//can optimize
	for (int i = 0; i < GEN_LEN_PAD; i++)
	{
    	for (int j = 0; j < GEN_LEN_PAD; j++)
		{
        	if (i >= j)  mask_base[i][j] = 0.0f;
        	else 		 mask_base[i][j] = -10000.0f;
    	}
	}

    //make the mask based on mask_base
    for(int b = 0; b < IN_BATCH; b++)
	{
		for (int orow = 0; orow < SEQ_LEN; orow++)
		{
        	for (int ocol = 0; ocol < SEQ_LEN; ocol++)
			{
            	mask_used[b][orow][ocol] = mask_base[orow][ocol];
                if(input_attn_mask[b][ocol] == 0) mask_used[b][orow][ocol] -= 10000.0f;
       	 	}
    	}
	}

    //MASKING
	for (int b = 0; b < IN_BATCH; b++)
	{
		for (int num = 0; num < NUM_HEAD; num++)
		{
			for (int orow = 0; orow < SEQ_LEN; orow++)
			{
				for (int ocol = 0; ocol < SEQ_LEN; ocol++)
				{
					attention_score[b][num][orow][ocol] += mask_used[b][orow][ocol];
				}
			}
		}
	}

    end = read_cycles();
    attn_masking_cycles += end - start;
    printf_v("ATTN - MASKING FINISHED\n");
}

//softmax
void attn_softmax(int IN_BATCH, int SEQ_LEN)
{   
    start = read_cycles();

    for (int b = 0; b < IN_BATCH; b++)
    {
        for (int num = 0; num < NUM_HEAD; num++)
        {
            for(int orow = 0; orow < SEQ_LEN; orow++)
            {
                //find the max_score
                float max_score = attention_score[b][num][orow][0];
                for (int ocol = 1; ocol < SEQ_LEN; ocol++)
                {
                    if (attention_score[b][num][orow][ocol] > max_score)
                    {
                        max_score = attention_score[b][num][orow][ocol];
                    }
                }

                //get sum_exp of stablized score
                float sum_exp = 0.0;
                for (int ocol = 0; ocol < SEQ_LEN; ocol++)
                {
                    attention_probs[b][num][orow][ocol] = (float)exp_riscv((double)(attention_score[b][num][orow][ocol] - max_score));
                    sum_exp += attention_probs[b][num][orow][ocol];
                }

                //get probs
                for (int ocol = 0; ocol < SEQ_LEN; ocol++)
                {
                    attention_probs[b][num][orow][ocol] /= sum_exp;
                }
            }
        }
    }

    end = read_cycles();
    attn_softmax_cycles += end - start;
    printf_v("ATTN - SOFTMAX FINISHED\n");
}

//attention final step
//attention prob * value matrix
void attn_ctx(int IN_BATCH, int SEQ_LEN)
{   
    start = read_cycles();

    for (int b = 0; b < IN_BATCH; b++)
	{
		for (int num = 0; num < NUM_HEAD; num++)
		{
			for (int row = 0; row < SEQ_LEN; row++)
			{
				for (int col = 0; col < SIZ_HEAD; col++)
				{
					float temp = 0.0f;

					for (int k = 0; k < SEQ_LEN; k++)
					{
						temp += attention_probs[b][num][row][k] * value_perm[b][num][k][col];
					}
					context_layer[b][num][row][col] = temp;
				}
			}
		}
	}

    end = read_cycles();
    attn_ctx_cycles += end - start;
    printf_v("ATTN - CONTEXT FINISHED\n");
}

//context resizing
//merge multi-head
void attn_ctx_reshape(int IN_BATCH, int SEQ_LEN)
{
    start = read_cycles();

    //permutation of context
    for (int b = 0; b < IN_BATCH; b++)
	{
		for (int seq = 0; seq < SEQ_LEN; seq++)
		{
			for (int num = 0; num < NUM_HEAD; num++)
			{
				for (int siz = 0; siz < SIZ_HEAD; siz++)
				{
					context_layer_perm[b][seq][num][siz] = context_layer[b][num][seq][siz];
				}
			}
		}
	}

	//merge multi-head
	for (int b = 0; b < IN_BATCH; b++)
	{
		for (int seq = 0; seq < SEQ_LEN; seq++)
		{
			for (int num = 0; num < NUM_HEAD; num++)
			{
				for (int siz = 0; siz < SIZ_HEAD; siz++)
				{
					context_layer_output[b][seq][num * SIZ_HEAD + siz] = context_layer_perm[b][seq][num][siz];
				}
			}
		}
	}

    end = read_cycles();
    attn_merge_cycles += end - start;
    printf_v("ATTN - MERGE FINISHED\n");
}

//projection in attention layer
void attn_proj(int layer, int IN_BATCH, int SEQ_LEN)
{   
    start = read_cycles();

    for (int b = 0; b < IN_BATCH; b++)
	{
		for (int i = 0; i < SEQ_LEN; i++)
		{
			for (int j = 0; j < HLEN; j++)
			{
				context_layer_proj[b][i][j] =  attn_proj_b[layer][j];

				for (int k = 0; k < HLEN; k++)
				{
					context_layer_proj[b][i][j] += context_layer_output[b][i][k] * attn_proj_w[layer][k][j];
				}
			}
		}
	}

    end = read_cycles();
    attn_proj_cycles += end - start;
    printf_v("ATTN - PROJ FINISHED\n");
}

//residual addition in attention layar
void attn_resadd(int IN_BATCH, int SEQ_LEN)
{   
    start = read_cycles();

    for (int b = 0; b < IN_BATCH; b++)
	{
		for (int i = 0; i < SEQ_LEN; i++)
		{
			for (int j = 0; j < HLEN; j++)
			{
				context_resadd[b][i][j] = context_layer_proj[b][i][j] + block_input[b][i][j];
			}
		}
	}

    end = read_cycles();
    attn_resadd_cycles += end - start;
    printf_v("ATTN - RESADD FINISHED\n");
}

//normalization - mlp layer
void mlp_norm(int layer, int IN_BATCH, int SEQ_LEN, float eps)
{   
    start = read_cycles();
    
    for (int b = 0; b < IN_BATCH; b++)
	{
		for (int seq = 0; seq < SEQ_LEN; seq++)
		{
			// calculate each mean, var based on each vector(HLEN)
			float avg = 0.0f;
			float var = 0.0f;
			float sum = 0.0f;
			float sqsum = 0.0f;

			for (int idx = 0; idx < HLEN; idx++)
			{
				sum   += context_resadd[b][seq][idx];
				sqsum += context_resadd[b][seq][idx] * context_resadd[b][seq][idx];
			}

			avg = sum / HLEN;
			var = sqsum / HLEN - avg * avg;

            //normalize
			for (int idx = 0; idx < HLEN; idx++)
			{
				input_mlp[b][seq][idx] = (context_resadd[b][seq][idx] - avg) / sqrtf(var + eps) * ln_2_w[layer][idx] + ln_2_b[layer][idx];
			}
		}
	}

    end = read_cycles();
    mlp_norm_cycles += end - start;
    printf_v("MLP - NORM FINISHED\n");
}

void mlp_fc(int layer, int IN_BATCH, int SEQ_LEN)
{
    start = read_cycles();

    for (int b = 0; b < IN_BATCH; b++)
	{
		for (int i = 0; i < SEQ_LEN; i++)
		{
			for (int j = 0; j < IMLEN; j++)
			{
				mlp_fc_output[b][i][j] = mlp_fc_b[layer][j];

				for (int k = 0; k < HLEN; k++)
				{
					mlp_fc_output[b][i][j] += input_mlp[b][i][k] * mlp_fc_w[layer][k][j];
				}
			}
		}
	}

    end = read_cycles();
    mlp_fc_cycles += end - start;
    printf_v("MLP - FC FINISHED\n");
}


//GELU
float gelu(float x)
{
	float ret = 0.5 * x * (1 + (float)tanh_riscv((double)(sqrt(2 / M_PI) * (x + 0.044715 * x * x * x))));
	//float ret = 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * x * x * x)));
	return ret;
}

void mlp_gelu_forward(int IN_BATCH, int SEQ_LEN)
{   
    start = read_cycles();

    for (int b = 0; b < IN_BATCH; b++)
	{
		for (int i = 0; i < SEQ_LEN; i++)
		{
			for (int j = 0; j < IMLEN; j++)
			{
				mlp_gelu[b][i][j] = gelu(mlp_fc_output[b][i][j]);
			}
		}
	}

    end = read_cycles();
    mlp_gelu_cycles += end - start;
    printf_v("MLP - GELU FINISHED\n");
}

//projection in mlp layer
void mlp_proj_forward(int layer, int IN_BATCH, int SEQ_LEN)
{   
    start = read_cycles();

    for (int b = 0; b < IN_BATCH; b++)
	{
		for (int i = 0; i < SEQ_LEN; i++)
		{
			for (int j = 0; j < HLEN; j++)
			{
				mlp_proj[b][i][j] = mlp_proj_b[layer][j];

				for (int k = 0; k < IMLEN; k++)
				{
					mlp_proj[b][i][j] += mlp_gelu[b][i][k] * mlp_proj_w[layer][k][j];
				}
			}
		}
	}

    end = read_cycles();
    mlp_proj_cycles += end - start;
    printf_v("MLP - PROJ FINISHED\n");
}

//Final residual add to get output of the transformer block
void mlp_resadd(int IN_BATCH, int SEQ_LEN)
{   
    start = read_cycles();

    for (int b = 0; b < IN_BATCH; b++)
	{
		for (int i = 0; i < SEQ_LEN; i++)
		{
			for (int j = 0; j < HLEN; j++)
			{
				block_output[b][i][j] = mlp_proj[b][i][j] + context_resadd[b][i][j];
			}
		}
	}

    end = read_cycles();
    mlp_resadd_cycles += end - start;
    printf_v("MLP - RESADD FINISHED\n");
}

/*
    MODEL FORWARD FUNCTIONS

    1. Transfer output to next input
    2. Forwarding an entire block
    3. Final normalization after passing entire transformer
    4. Getting Logits by matrix multiplication
    5. Stablized softmax of logits
    6. Argmax of probablity
	7. Fowarding Entire model 

*/

void output_to_input(int IN_BATCH, int SEQ_LEN, int flag);
void block_forward(int layer, int IN_BATCH, int SEQ_LEN);
void model_norm(int IN_BATCH, int SEQ_LEN, float eps);
void model_logits(int IN_BATCH, int SEQ_LEN);
void model_softmax(int IN_BATCH, int SEQ_LEN); 
void model_argmax(int IN_BATCH, int SEQ_LEN); 
void model_forward(int IN_BATCH, int SEQ_LEN, int flag);


//assign block output to the next block input
//flag=START_GPT2  : block_input_init -> block_input
//flag=MOVE_GPT2   : block_output     -> block_input
//flag=FINISH_GPT2 : block_output     -> block_output_last
void output_to_input(int IN_BATCH, int SEQ_LEN, int flag)
{
    for(int b = 0; b < IN_BATCH; b++)
    {
        for(int seq = 0; seq < GEN_LEN_PAD; seq++)
        {
            for(int idx = 0; idx < HLEN; idx++)
            {
                if      (flag == START_GPT2)  block_input[b][seq][idx]        = block_input_init[b][seq][idx];
                else if (flag == FINISH_GPT2) block_output_last[b][seq][idx]  = block_output[b][seq][idx];
                else                          block_input[b][seq][idx]        = block_output[b][seq][idx];
            }
        }
    }
}

void block_forward(int layer, int IN_BATCH, int SEQ_LEN)                                                             
{       
	printf_v("\n<LAYER %d>\n\n", layer);                                                                                                            
    attn_norm(layer, IN_BATCH, SEQ_LEN, EPS);
    attn_QKV(layer, IN_BATCH, SEQ_LEN);
    attn_reshape(IN_BATCH, SEQ_LEN);
    attn_score(IN_BATCH, SEQ_LEN);      
    attn_masking(IN_BATCH, SEQ_LEN); 
    attn_softmax(IN_BATCH, SEQ_LEN);     
    attn_ctx(IN_BATCH, SEQ_LEN);
    attn_ctx_reshape(IN_BATCH, SEQ_LEN); 
    attn_proj(layer, IN_BATCH, SEQ_LEN);
    attn_resadd(IN_BATCH, SEQ_LEN);
    mlp_norm(layer, IN_BATCH, SEQ_LEN, EPS);  
    mlp_fc(layer, IN_BATCH, SEQ_LEN); 
    mlp_gelu_forward(IN_BATCH, SEQ_LEN);  
    mlp_proj_forward(layer, IN_BATCH, SEQ_LEN);
    mlp_resadd(IN_BATCH, SEQ_LEN);
	print_cycles_info();
}


//normalize transformer output
void model_norm(int IN_BATCH, int SEQ_LEN, float eps)
{   
    start = read_cycles();

    for (int b = 0; b < IN_BATCH; b++)
	{
		for (int seq = 0; seq < SEQ_LEN; seq++)
		{
			// calculate each mean, var based on each vector(HLEN)
			float avg = 0.0f;
			float var = 0.0f;
			float sum = 0.0f;
			float sqsum = 0.0f;

			for (int idx = 0; idx < HLEN; idx++)
			{
				sum   += block_output_last[b][seq][idx];
				sqsum += block_output_last[b][seq][idx] * block_output_last[b][seq][idx];
			}

			avg = sum / HLEN;
			var = sqsum / HLEN - avg * avg;

            //normalize
			for (int idx = 0; idx < HLEN; idx++)
			{
				output_norm[b][seq][idx] = (block_output_last[b][seq][idx] - avg) / sqrtf(var + eps) * ln_f_w[idx] + ln_f_b[idx];
			}
		}
	}

    end = read_cycles();
    model_norm_cycles += end - start;
    printf_v("MODEL - NORM FINISHED\n");
}

//normalized output * wte_transpose
void model_logits(int IN_BATCH, int SEQ_LEN)
{   
    start = read_cycles();

    for (int b = 0; b < IN_BATCH; b++)
	{
		for (int s = 0; s < SEQ_LEN; s++)
		{
			for (int v = 0; v < VOCAB; v++)
			{
				output_logits[b][s][v] = 0.0f;

				for (int k = 0; k < HLEN; k++)
				{
					output_logits[b][s][v] += output_norm[b][s][k] * wte[v][k];
				}
			}
		}
	}

    end = read_cycles();
    model_logits_cycles += end - start;
    printf_v("MODEL - LOGITS FINISHED\n");
}

//softmax to get output probability
void model_softmax(int IN_BATCH, int SEQ_LEN)
{   
    start = read_cycles();

    for (int b = 0; b < IN_BATCH; b++)
    {
        for (int s = SEQ_LEN - 1; s < SEQ_LEN; s++)
        {
            //find the max logit
            float max_logit = output_logits[b][s][0];
            for (int v = 1; v < VOCAB; v++)
            {
                if (output_logits[b][s][v] > max_logit)
                {
                    max_logit = output_logits[b][s][v];
                }
            }

            //get sum_exp of stablized logits
            float sum_exp = 0.0;
            for (int v = 0; v < VOCAB; v++)
            {
				output_probs[b][s][v] = (float)exp_riscv((double)(output_logits[b][s][v] - max_logit));
                sum_exp += output_probs[b][s][v];
            }

            //get probs
            for (int v = 0; v < VOCAB; v++)
            {
                output_probs[b][s][v] /= sum_exp;
            }
        }
    }

    end = read_cycles();
    model_softmax_cycles += end - start;
    printf_v("MODEL - SOFTMAX FINISHED\n");
}

//argmax to get the prediction of next token
void model_argmax(int IN_BATCH, int SEQ_LEN)
{   
    start = read_cycles();

    for (int b = 0; b < IN_BATCH; b++)
    {
        for (int s = SEQ_LEN - 1; s < SEQ_LEN; s++)
        {
            int max_index = 0;
            float max_value = output_probs[b][s][0];

            //find max_value & max_index
            for (int v = 1; v < VOCAB; v++)
            {
                if (output_probs[b][s][v] > max_value) {
                    max_value = output_probs[b][s][v];
                    max_index = v;
                }
            }
            output_token_ids[b][s] = max_index;
        }
    }

    end = read_cycles();
    model_argmax_cycles += end - start;
    printf_v("MODEL - ARGMAX FINISHED\n");
}

void model_forward(int IN_BATCH, int SEQ_LEN, int flag)                                                              
{                                                                                                 
    //Embedding vectors                                                                                              
    allocate_model_layer_input(IN_BATCH, SEQ_LEN, flag);                                                             
                                                                                                                     
    //Transformer                                                                                                    
    for(int layer = 0; layer < LAYER; layer++)                                                                       
    {                                                                                                                
        output_to_input(IN_BATCH, SEQ_LEN, (layer > 0) ? MOVE_GPT2 : START_GPT2);                                    
        block_forward(layer, IN_BATCH, SEQ_LEN);                                                                     
    }                                                                                                                
                                                                                                                     
    output_to_input(IN_BATCH, SEQ_LEN, FINISH_GPT2);                                                                 
    model_norm(IN_BATCH, SEQ_LEN, EPS);                                                                              
                                                                                                                     
    //Token prediction
    model_logits(IN_BATCH, SEQ_LEN);                                                                                 
    model_softmax(IN_BATCH, SEQ_LEN);                                                                                
    model_argmax(IN_BATCH, SEQ_LEN);                                                                                 
}

/*
    FUNCTIONS for GEMMINI(MATMUL, RESADD)
*/

void attn_QKV_gemmini(int layer, int IN_BATCH, int SEQ_LEN, int flag);
void attn_score_gemmini(int IN_BATCH, int SEQ_LEN);
void attn_ctx_gemmini(int IN_BATCH, int SEQ_LEN);
void attn_proj_gemmini(int layer, int IN_BATCH, int SEQ_LEN);
void attn_resadd_gemmini(int IN_BATCH, int SEQ_LEN);
void mlp_fc_gemmini(int layer, int IN_BATCH, int SEQ_LEN);
void mlp_proj_gemmini(int layer, int IN_BATCH, int SEQ_LEN);
void mlp_resadd_gemmini(int IN_BATCH, int SEQ_LEN);
void model_logits_gemmini(int IN_BATCH, int SEQ_LEN);
void block_forward_gemmini(int layer, int IN_BATCH, int SEQ_LEN, int flag); 
void model_forward_gemmini(int IN_BATCH, int SEQ_LEN, int flag);

void attn_QKV_gemmini(int layer, int IN_BATCH, int SEQ_LEN, int flag)
{
	start = read_cycles();

	//move last input row into temporary array 
	if(flag == USE_CACHE)
	{
		for (int b = 0; b < IN_BATCH; b++)
		{
			for(int i = 0; i < HLEN; i++)
			{
				last_input_norm[b][0][i] = input_norm[b][SEQ_LEN - 1][i];
			}
		}
	}

	for (int b = 0; b < IN_BATCH; b++)
	{

		tiled_matmul_nn_auto(GET_DIM_I(SEQ_LEN), HLEN, HLEN,
							 input_norm[b], attn_query_w[layer], attn_query_b[layer], query_layer[b],
							 NO_ACTIVATION, 1, true,
							 tiled_matmul_type, check, "attention-Q");
		if(flag == USE_CACHE)
		{
			tiled_matmul_nn_auto(GET_DIM_I(1), HLEN, HLEN,
								last_input_norm[b], attn_key_w[layer], attn_key_b[layer], last_key_layer[b],
								NO_ACTIVATION, 1, true,
								tiled_matmul_type, check, "attention-K");
			
			tiled_matmul_nn_auto(GET_DIM_I(1), HLEN, HLEN,
								last_input_norm[b], attn_value_w[layer], attn_value_b[layer], last_value_layer[b],
								NO_ACTIVATION, 1, true,
								tiled_matmul_type, check, "attention-V");	
		}
		else
		{
			tiled_matmul_nn_auto(GET_DIM_I(SEQ_LEN), HLEN, HLEN,
							 input_norm[b], attn_key_w[layer], attn_key_b[layer], key_layer[b],
							 NO_ACTIVATION, 1, true,
							 tiled_matmul_type, check, "attention-K");
		
			tiled_matmul_nn_auto(GET_DIM_I(SEQ_LEN), HLEN, HLEN,
							 input_norm[b], attn_value_w[layer], attn_value_b[layer], value_layer[b],
							 NO_ACTIVATION, 1, true,
							 tiled_matmul_type, check, "attention-V");
		}
    }

	//move to KV Cache
	int start_idx = flag == USE_CACHE ? SEQ_LEN - 1 : 0;

	for(int b = 0; b < IN_BATCH; b++)
	{
		for(int seq = start_idx; seq < SEQ_LEN; seq++)
		{
			for(int i = 0; i < HLEN; i++)
			{
				KV_Cache[layer][KV_K][b][seq][i] = flag == USE_CACHE ? last_key_layer[b][0][i]   : key_layer[b][seq][i];
				KV_Cache[layer][KV_V][b][seq][i] = flag == USE_CACHE ? last_value_layer[b][0][i] : value_layer[b][seq][i];
			}
		}
	}

	//Get KV matrix
	if(flag == USE_CACHE)
	{
		for(int b = 0; b < IN_BATCH; b++)
		{
			for(int seq = 0; seq < SEQ_LEN; seq++)
			{
				for(int i = 0; i < HLEN; i++)
				{
					key_layer[b][seq][i]   = KV_Cache[layer][KV_K][b][seq][i];
					value_layer[b][seq][i] = KV_Cache[layer][KV_V][b][seq][i];
				}
			}
		}
	}
	
	end = read_cycles();

	attn_QKV_cycles += end - start;

	printf_v("ATTN - QKV FINISHED\n");
}

void attn_score_gemmini(int IN_BATCH, int SEQ_LEN)
{
    start = read_cycles();

    for (int b = 0; b < IN_BATCH; b++)
    {
        for (int num = 0; num < NUM_HEAD; num++)
        {
            
			//due to storing policy, SEQ_LEN -> GEN_LEN_PAD
			tiled_matmul_nn_auto(GET_DIM_I(SEQ_LEN), GEN_LEN_PAD, SIZ_HEAD, 
								 query_perm[b][num], key_perm_t[b][num], NULL, attention_score[b][num],
								 NO_ACTIVATION, 1, true,
								 tiled_matmul_type, check, "attention-score");
			
			//scale
			for (int i = 0; i < SEQ_LEN; i++)
			{
				for (int j = 0; j < SEQ_LEN; j++)
				{
					attention_score[b][num][i][j] = attention_score[b][num][i][j] / sqrt(SIZ_HEAD);
				}
			}
        }
    }

    end = read_cycles();

    attn_score_cycles += end - start;

	printf_v("ATTN - SCORE FIINISHED\n");
}

void attn_ctx_gemmini(int IN_BATCH, int SEQ_LEN)
{
    start = read_cycles();

    for (int b = 0; b < IN_BATCH; b++)
    {
        for (int num = 0; num < NUM_HEAD; num++)
        {
			tiled_matmul_nn_auto(GET_DIM_I(SEQ_LEN), SIZ_HEAD, GEN_LEN_PAD,
								attention_probs[b][num], value_perm[b][num], NULL, context_layer[b][num],
								NO_ACTIVATION, 1, true,
								tiled_matmul_type, check, "attention-final");

        }
    }

    end = read_cycles();

    attn_ctx_cycles += end - start;

	printf_v("ATTN - CONTEXT FINISHED\n");
}

void attn_proj_gemmini(int layer, int IN_BATCH, int SEQ_LEN)
{	
    start = read_cycles();

	for(int b = 0; b < IN_BATCH; b++)
	{
		tiled_matmul_nn_auto(GET_DIM_I(SEQ_LEN), HLEN, HLEN,
							 context_layer_output[b], attn_proj_w[layer], attn_proj_b[layer], context_layer_proj[b],
							 NO_ACTIVATION, 1, true,
							 tiled_matmul_type, check, "attention-proj");
	}

    end = read_cycles();
    attn_proj_cycles += end - start;
	printf_v("ATTN - PROJ FINISHED\n");
}

void attn_resadd_gemmini(int IN_BATCH, int SEQ_LEN)
{
    start = read_cycles();

	for (int b = 0; b < IN_BATCH; b++)
	{
		tiled_resadd_auto(GET_DIM_I(SEQ_LEN), HLEN, 1, 1, 1, 
						  context_layer_proj[b][0], block_input[b][0], context_resadd[b][0], 
						  false,  tiled_matmul_type == CPU ? CPU : WS);
	}
	
    end = read_cycles();
    attn_resadd_cycles += end - start;
	printf_v("ATTN - RESADD FINISHED\n");
}

void mlp_fc_gemmini(int layer, int IN_BATCH, int SEQ_LEN)
{
    start = read_cycles();

    for (int b = 0; b < IN_BATCH; b++)
    {	
		tiled_matmul_nn_auto(GET_DIM_I(SEQ_LEN), IMLEN, HLEN,
							 input_mlp[b], mlp_fc_w[layer], mlp_fc_b[layer], mlp_fc_output[b],
							 NO_ACTIVATION, 1, true,
							 tiled_matmul_type, false, "mlp-fc"); //not able to check due to demand of big stack size for comparsion
	}

    end = read_cycles();
    mlp_fc_cycles += end - start;
	printf_v("MLP - FC FINISHED\n");
}

void mlp_proj_gemmini(int layer, int IN_BATCH, int SEQ_LEN)
{
    start = read_cycles();

	for (int b = 0; b < IN_BATCH; b++)
	{
		tiled_matmul_nn_auto(GET_DIM_I(SEQ_LEN), HLEN, IMLEN,
							 mlp_gelu[b], mlp_proj_w[layer], mlp_proj_b[layer], mlp_proj[b],
							 NO_ACTIVATION, 1, true,
							 tiled_matmul_type, check, "mlp-proj");
	}

    end = read_cycles();
    mlp_proj_cycles += end - start;
	printf_v("MLP - PROJ FINISHED\n");
}

void mlp_resadd_gemmini(int IN_BATCH, int SEQ_LEN)
{
    start = read_cycles();

	for (int b = 0; b < IN_BATCH; b++)
	{
		tiled_resadd_auto(GET_DIM_I(SEQ_LEN), HLEN, 1, 1, 1, 
						  context_resadd[b][0], mlp_proj[b][0], block_output[b][0], 
						  false, tiled_matmul_type == CPU ? CPU : WS);
	}

    end = read_cycles();
    mlp_resadd_cycles += end - start;
	printf_v("MLP - RESADD FINISHED\n");
}

void model_logits_gemmini(int IN_BATCH, int SEQ_LEN)
{   
    start = read_cycles();

	for (int b = 0; b < IN_BATCH; b++)
	{
		tiled_matmul_nn_auto(GET_DIM_I(SEQ_LEN), VOCAB_PAD, HLEN,
							 output_norm[b], wte_t, NULL, output_logits[b],
							 NO_ACTIVATION, 1, true,
							 tiled_matmul_type, false, "output-logits");
	}
	
    end = read_cycles();
    model_logits_cycles += end - start;
	printf_v("MODEL - LOGITS FINISHED\n");
}

void model_logits_gemmini_acc(int IN_BATCH, int SEQ_LEN)
{
	start = read_cycles();

	for (int b = 0; b < IN_BATCH; b++)
	{
		for(int i = 0; i < HLEN; i++)
		{
			last_output_norm[b][0][i] = output_norm[b][SEQ_LEN - 1][i];
		}
	}

	for (int b = 0; b < IN_BATCH; b++)
	{
		tiled_matmul_nn_auto(GET_DIM_I(1), VOCAB_PAD, HLEN,
							 last_output_norm[b], wte_t, NULL, last_output_logits[b],
							 NO_ACTIVATION, 1, true,
							 tiled_matmul_type, false, "output-logits");
	}

	for (int b = 0; b < IN_BATCH; b++)
	{
		for(int i = 0; i < VOCAB_PAD; i++)
		{
			output_logits[b][SEQ_LEN - 1][i] = last_output_logits[b][0][i];
		}
	}

	end = read_cycles();
    model_logits_cycles += end - start;
	printf_v("MODEL - LOGITS FINISHED\n");
}

void block_forward_gemmini(int layer, int IN_BATCH, int SEQ_LEN, int flag)                                                             
{   
	printf_v("\n<LAYER %d>\n\n", layer);                                                                                                   
    attn_norm(layer, IN_BATCH, SEQ_LEN, EPS);
   	attn_QKV_gemmini(layer, IN_BATCH, SEQ_LEN, flag);
    attn_reshape(IN_BATCH, SEQ_LEN);
    attn_score_gemmini(IN_BATCH, SEQ_LEN);      
    attn_masking(IN_BATCH, SEQ_LEN); 
    attn_softmax(IN_BATCH, SEQ_LEN);     
	attn_ctx_gemmini(IN_BATCH, SEQ_LEN);
    attn_ctx_reshape(IN_BATCH, SEQ_LEN); 
	attn_proj_gemmini(layer, IN_BATCH, SEQ_LEN);
	attn_resadd_gemmini(IN_BATCH, SEQ_LEN);
    mlp_norm(layer, IN_BATCH, SEQ_LEN, EPS);  
	mlp_fc_gemmini(layer, IN_BATCH, SEQ_LEN);
    mlp_gelu_forward(IN_BATCH, SEQ_LEN);  
	mlp_proj_gemmini(layer, IN_BATCH, SEQ_LEN);
	mlp_resadd_gemmini(IN_BATCH, SEQ_LEN);
	if(verbose) print_cycles_info();
}

void model_forward_gemmini(int IN_BATCH, int SEQ_LEN, int flag)                                                              
{                                                                                                                    
    //Embedding vectors                                                                                              
    allocate_model_layer_input(IN_BATCH, SEQ_LEN, flag);                                                             
                                                                                                                     
    //Transformer                                                                                                    
    for(int layer = 0; layer < LAYER; layer++)                                                                       
    {                                                                                                                
        output_to_input(IN_BATCH, SEQ_LEN, (layer > 0) ? MOVE_GPT2 : START_GPT2);                                    
        block_forward_gemmini(layer, IN_BATCH, SEQ_LEN, flag);                                                                     
    }                                                                                                                
                                                                                                                     
    output_to_input(IN_BATCH, SEQ_LEN, FINISH_GPT2);                                                                 
    model_norm(IN_BATCH, SEQ_LEN, EPS);                                                                              
                                                                                                                     
    //Token prediction
    //model_logits_gemmini(IN_BATCH, SEQ_LEN);
	model_logits_gemmini_acc(IN_BATCH, SEQ_LEN);                                                                                 
    model_softmax(IN_BATCH, SEQ_LEN);                                                                                
    model_argmax(IN_BATCH, SEQ_LEN);  
	
	if(verbose) print_cycles_info();                                                                               
}

#endif //GPT2_H
