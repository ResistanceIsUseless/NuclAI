#!/bin/bash
echo "Starting Nuclei RAG interactive mode..."
python nuclei_rag_system.py --interactive --index-path indices/nuclei_rag_index.pkl
