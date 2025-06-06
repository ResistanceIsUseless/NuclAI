#!/bin/bash
echo "Building Nuclei RAG index..."
python nuclei_rag_system.py --templates-path ~/nuclei-templates --build-index --index-path indices/nuclei_rag_index.pkl
echo "Index built successfully!"
