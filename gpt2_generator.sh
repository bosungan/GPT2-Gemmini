#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 [baremetal|pk [matmul_option [verbose]]]"
    echo "  - baremetal: Compile and run gpt2-baremetal"
    echo "  - pk: Compile and run gpt2-pk with optional matmul_option and verbose"
    echo "  - matmul_option: One of 'os', 'ws', or 'cpu' (only for pk)"
    echo "  - verbose: Enable verbose mode (only for pk)"
    exit 1
}

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
    usage
fi

# Determine the mode and options
mode="$1"
shift

case "$mode" in
    baremetal)
        # Run encoder
        python ./src/encoder.py

        # Recompile gpt2.elf
        cd src
        if [ -f "gpt2-baremetal.o" ]; then
            rm gpt2-baremetal.o
        fi
        make baremetal -s 2>/dev/null

        # Execute the gpt2-baremetal program & write output token ids
        cd ..
        stdbuf -oL spike --extension=gemmini ./src/gpt2-baremetal | tee >(grep "key" > ./output/generated_token_ids.txt) | grep -v "key"

        # Run decoder
        python ./src/decoder.py
        ;;
    pk)
        # Run encoder
        python ./src/encoder.py

        # Recompile gpt2.elf
        cd src
        if [ -f "gpt2-pk.o" ]; then
            rm gpt2-pk.o
        fi
        make pk -s 2>/dev/null

        # Process optional arguments for pk
        matmul_option=""
        verbose_option=""
        
        if [ "$#" -ge 1 ]; then
            matmul_option="$1"
            shift
        fi

        if [ "$#" -ge 1 ] && [ "$1" == "verbose" ]; then
            verbose_option="verbose"
        fi

        # Execute the gpt2-pk program with options
        cd ..
        if [ -z "$verbose_option" ]; then
            unbuffer spike --extension=gemmini pk ./src/gpt2-pk $matmul_option | tee >(grep "key" > ./output/generated_token_ids.txt) | grep -v "key"
        else
            unbuffer spike --extension=gemmini pk ./src/gpt2-pk $matmul_option $verbose_option | tee >(grep "key" > ./output/generated_token_ids.txt) | grep -v "key"
        fi

        # Run decoder
        python ./src/decoder.py
        ;;
    *)
        usage
        ;;
esac
