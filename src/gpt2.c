#include <stdio.h>
#include <stdlib.h>
#include "gpt2.h"

/*
    version 6.0

	Supporting KV cache

*/
enum tiled_matmul_type_t tiled_matmul_type = WS;
bool check = false; // cannout use check function due to matmul_cpu requirments(dim_I % 4 == 0, dim_J % 4 == 0)
bool verbose = false;


int main(int argc, char* argv[])
{   
    if (argc < 2) {
        tiled_matmul_type = OS;
    } else if (strcmp(argv[1], "cpu") == 0) {
        tiled_matmul_type = CPU;
    } else if (strcmp(argv[1], "os") == 0) {
        tiled_matmul_type = OS;
    } else if (strcmp(argv[1], "ws") == 0) {
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "-h") == 0) {
        printf("usage: %s [-h] matmul_option [verbose]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        printf("  In verbose mode, you can see every process of inference and taken cycles info after fowarding every layer\n");
        exit(0);
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [verbose]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        printf(". In verbose mode, you can see every process of inference and taken cycles info after fowarding every layer\n");
        exit(1);
    }

    if (argc < 3) {
        verbose = false;
    } else if (strcmp(argv[2], "verbose") == 0) {
        verbose = true;
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [verbose]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        printf("  In verbose mode, you can see every process of inference and taken cycles info after fowarding every layer\n");
        exit(1);
    }

    gemmini_flush(0);

    //generate token ids seqeunce 
    printf("\n\n\n\t\tINFERENCE START\n\n\n");
    printf("Batch size: %d, Input length: %d, Target length: %d\n\n", IN_BATCH_INIT, SEQ_LEN_INIT, GEN_LEN);
    printf("-------------------------------------------------------------------------\n");

	//prepare wte_t
	TRANSPOSE_MATRIX(wte, wte_t, VOCAB, HLEN);

    //main code 
    int SEQ_LEN = SEQ_LEN_INIT;

    //generate
    for(int gen_len = SEQ_LEN_INIT; gen_len < GEN_LEN; gen_len++)
    {   

        //forward entire model
		//model_forward(IN_BATCH_INIT, SEQ_LEN, (gen_len == SEQ_LEN_INIT) ? USE_NONE : USE_CACHE);
        model_forward_gemmini(IN_BATCH_INIT, SEQ_LEN, (gen_len == SEQ_LEN_INIT) ? USE_NONE : USE_CACHE);

        //print status
        for (int b = 0; b < IN_BATCH_INIT; b++) {

            if(b == 0){
                printf("[FORWARD %2d] : COMPLETE   | (ADDED TOKEN ID FOR BATCH %d: %5d) | %3d/%3d\n",gen_len - SEQ_LEN_INIT + 1, b, output_token_ids[b][gen_len - 1], gen_len - SEQ_LEN_INIT + 1, GEN_LEN - SEQ_LEN_INIT);
            }

            else{
                printf("                          | (ADDED TOKEN ID FOR BATCH %d: %5d) |\n", b, output_token_ids[b][gen_len - 1]);
            }
        }
        printf("-------------------------------------------------------------------------\n");


        //append the last predicted token 
        for(int b = 0; b < IN_BATCH_INIT; b++) 
        {
           input_token_ids[b][gen_len] = output_token_ids[b][gen_len - 1];
           input_attn_mask[b][gen_len] = 1; 
        }

        SEQ_LEN++; 
    }

    //END
    printf("\n\n\n\t\tINFERENCE FINISHED\n\n\n");
    print_cycles_info();

	//Print the results of Inference
	//we will make the .txt file of result by grep & tee
	printf("key %d\n", IN_BATCH_INIT);
	printf("key %d\n", GEN_LEN);

	for(int b = 0; b  < IN_BATCH_INIT; b++)
	{
		for(int gen_len = 0; gen_len < GEN_LEN; gen_len++)
		{
			printf("key %d\n", input_token_ids[b][gen_len]);
		}
	}
}
