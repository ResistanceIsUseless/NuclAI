#!/bin/bash
if [ -z "$1" ]; then
    echo "Usage: ./search_templates.sh 'your search query'"
    exit 1
fi

python nuclei_rag_system.py --query "$1" --search-only --index-path indices/nuclei_rag_index.pkl
