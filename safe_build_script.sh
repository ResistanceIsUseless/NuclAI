#!/bin/bash
# Safe build script for M4 Mac - handles memory issues

set -e

echo "=== Building Nuclei RAG Index (M4 Safe Mode) ==="

# Create indices directory
mkdir -p indices

# Check if we should use simple search mode
USE_SIMPLE=""
if [[ "$1" == "--simple" ]]; then
    USE_SIMPLE="--simple-search"
    echo "Using simple search mode (no semantic embeddings)"
fi

# Build index with error handling
echo "Building RAG index..."

python3 -c "
import sys
import os
sys.path.append('.')

try:
    from robust_nuclei_rag import RobustNucleiRAG
    
    # Use simple search mode to avoid FAISS issues
    rag = RobustNucleiRAG('~/nuclei-templates', use_simple_search=True)
    
    print('Loading templates...')
    rag.load_templates()
    
    print('Building search index...')
    rag.build_search_index()
    
    print('Saving index...')
    rag.save_index('indices/nuclei_rag_index.pkl')
    
    print('✅ Index built successfully!')
    print(f'Templates loaded: {len(rag.templates)}')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

# Verify the index was created
if [ -f "indices/nuclei_rag_index.pkl" ]; then
    echo "✅ Index file created successfully"
    
    # Test the index
    echo "Testing index..."
    python3 -c "
from robust_nuclei_rag import RobustNucleiRAG
rag = RobustNucleiRAG()
rag.load_index('indices/nuclei_rag_index.pkl')
results = rag.search('SQL injection', 3)
print(f'Test search found {len(results)} results')
for template, score in results:
    print(f'- {template.name} (Score: {score:.3f})')
"
    
    echo "✅ Index test passed!"
else
    echo "❌ Index file not created"
    exit 1
fi

echo ""
echo "=== RAG System Ready! ==="
echo "Try these commands:"
echo "  python3 robust_nuclei_rag.py --query 'SQL injection' --search-only"
echo "  python3 robust_nuclei_rag.py --interactive"
echo "  python3 robust_nuclei_rag.py --query 'XSS vulnerability' --model-path /path/to/model.gguf"
