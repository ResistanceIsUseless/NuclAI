#!/bin/bash
if [ -z "$1" ]; then
    echo "Usage: ./generate_template.sh 'template description'"
    exit 1
fi

MODEL_PATH=""
if [ -n "$2" ]; then
    MODEL_PATH="--model-path $2"
fi

python nuclei_rag_system.py --query "$1" $MODEL_PATH --index-path indices/nuclei_rag_index.pkl
