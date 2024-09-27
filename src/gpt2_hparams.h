#ifndef GPT2_HPARAMS_H
#define GPT2_HPARAMS_H

//MODEL INFO 
#define LAYER     12    // # of transformer block
#define SEQ_MAX   1024  // MAX length of input sequence 
#define HLEN      768   // hidden layer dimension
#define NUM_HEAD  12	// num_attention_heads
#define SIZ_HEAD  64    // attention_head_size
#define IMLEN     3072  // HLEN * 4 = IMLEN (MLP)
#define VOCAB     50257 // # of vocab table entry
#define VOCAB_PAD 50272 // multiple of 16
#define PAD_ID    50256 // <PAD> = <EOS>

//MODEL DETAILS
#define NO_LAYER    -1     // Macro for parameter used in weight loading
#define START_GPT2   1     // Macro for parameter used in transfer data
#define MOVE_GPT2    0     // Macro for parameter used in transfer data
#define FINISH_GPT2 -1     // Macro for parameter used in transfer data 
#define USE_CACHE    1     // Macro for parameter used in layer allocation
#define USE_NONE     0     // Macro for parameter used in layer allocation 
#define EPS          1e-5f // epsilon value for normalization
#define KV_NUM       2     // Macro for KV cache
#define KV_K         0     // KV cache - K
#define KV_V         1     // KV cache - V

#endif //GPT2_HPARAMS_H
