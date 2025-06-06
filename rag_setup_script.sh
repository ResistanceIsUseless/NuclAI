#!/bin/bash
# Nuclei RAG System Setup Script

set -e

echo "=== Nuclei Template RAG System Setup ==="

# =============================================================================
# PART 1: Install Dependencies
# =============================================================================

echo "Step 1: Installing Python dependencies..."

# Create requirements file
cat > requirements_rag.txt << 'EOF'
# Core RAG dependencies
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
numpy>=1.21.0
scikit-learn>=1.0.0
PyYAML>=6.0

# Optional: For local LLM
llama-cpp-python>=0.2.0

# Utilities
tqdm>=4.64.0
pathlib
EOF

# Install dependencies
pip install -r requirements_rag.txt

echo "✅ Dependencies installed"

# =============================================================================
# PART 2: Setup Directory Structure
# =============================================================================

echo "Step 2: Setting up directory structure..."

mkdir -p {data,models,indices,examples,logs}

echo "✅ Directory structure created"

# =============================================================================
# PART 3: Create Usage Examples
# =============================================================================

echo "Step 3: Creating usage examples..."

# Quick start script
cat > quick_start_rag.py << 'EOF'
#!/usr/bin/env python3
"""
Quick start example for Nuclei RAG system
"""

from nuclei_rag_system import NucleiRAGSystem, create_llm_function

def main():
    print("=== Nuclei RAG Quick Start ===")
    
    # Initialize RAG system
    rag = NucleiRAGSystem("~/nuclei-templates")
    
    # Build index (first time only)
    print("Loading templates and building index...")
    rag.load_templates()
    rag.build_vector_index()
    rag.build_keyword_index()
    rag.save_index("indices/nuclei_rag_index.pkl")
    
    # Test search
    query = "SQL injection in login forms"
    print(f"\nSearching for: {query}")
    
    results = rag.hybrid_search(query, 3)
    
    print(f"Found {len(results)} relevant templates:")
    for template, score in results:
        print(f"- {template.name} (Score: {score:.3f})")
        print(f"  Category: {template.category}")
        print(f"  Description: {template.description[:100]}...")
        print()
    
    # Generate new template
    print("Generating new template...")
    llm_function = create_llm_function()  # Mock function for demo
    
    new_template = rag.generate_template_with_rag(query, llm_function)
    print("Generated Template:")
    print("=" * 50)
    print(new_template)

if __name__ == "__main__":
    main()
EOF

# Test search script
cat > test_search.py << 'EOF'
#!/usr/bin/env python3
"""
Test different search queries
"""

from nuclei_rag_system import NucleiRAGSystem

def test_searches():
    # Load existing index
    rag = NucleiRAGSystem("~/nuclei-templates")
    rag.load_index("indices/nuclei_rag_index.pkl")
    
    test_queries = [
        "SQL injection vulnerability",
        "Cross-site scripting XSS",
        "Apache Log4j remote code execution",
        "WordPress plugin vulnerability",
        "Directory traversal path traversal",
        "Authentication bypass",
        "File upload vulnerability",
        "Command injection"
    ]
    
    for query in test_queries:
        print(f"\n=== Query: {query} ===")
        results = rag.hybrid_search(query, 3)
        
        for i, (template, score) in enumerate(results, 1):
            print(f"{i}. {template.name} (Score: {score:.3f})")
            print(f"   Category: {template.category} | Severity: {template.severity}")

if __name__ == "__main__":
    test_searches()
EOF

chmod +x quick_start_rag.py test_search.py

echo "✅ Example scripts created"

# =============================================================================
# PART 4: Create Configuration
# =============================================================================

echo "Step 4: Creating configuration..."

cat > rag_config.json << 'EOF'
{
  "templates_path": "~/nuclei-templates",
  "index_path": "indices/nuclei_rag_index.pkl",
  "embedding_model": "all-MiniLM-L6-v2",
  "search_settings": {
    "semantic_weight": 0.7,
    "keyword_weight": 0.3,
    "top_k_default": 5,
    "min_similarity_threshold": 0.1
  },
  "generation_settings": {
    "max_examples": 3,
    "temperature": 0.1,
    "max_tokens": 2000
  },
  "llm_settings": {
    "model_path": null,
    "n_ctx": 4096,
    "n_threads": 8
  }
}
EOF

echo "✅ Configuration created"

# =============================================================================
# PART 5: Create Wrapper Scripts
# =============================================================================

echo "Step 5: Creating wrapper scripts..."

# Build index script
cat > build_index.sh << 'EOF'
#!/bin/bash
echo "Building Nuclei RAG index..."
python nuclei_rag_system.py --templates-path ~/nuclei-templates --build-index --index-path indices/nuclei_rag_index.pkl
echo "Index built successfully!"
EOF

# Interactive mode script
cat > interactive_rag.sh << 'EOF'
#!/bin/bash
echo "Starting Nuclei RAG interactive mode..."
python nuclei_rag_system.py --interactive --index-path indices/nuclei_rag_index.pkl
EOF

# Search only script
cat > search_templates.sh << 'EOF'
#!/bin/bash
if [ -z "$1" ]; then
    echo "Usage: ./search_templates.sh 'your search query'"
    exit 1
fi

python nuclei_rag_system.py --query "$1" --search-only --index-path indices/nuclei_rag_index.pkl
EOF

# Generate template script
cat > generate_template.sh << 'EOF'
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
EOF

chmod +x build_index.sh interactive_rag.sh search_templates.sh generate_template.sh

echo "✅ Wrapper scripts created"

# =============================================================================
# PART 6: Create Documentation
# =============================================================================

echo "Step 6: Creating documentation..."

cat > RAG_USAGE.md << 'EOF'
# Nuclei Template RAG System Usage Guide

## Quick Start

### 1. Build Index (First Time Only)
```bash
./build_index.sh
```

### 2. Search Templates
```bash
./search_templates.sh "SQL injection"
```

### 3. Generate Template (Mock)
```bash
./generate_template.sh "XSS in contact forms"
```

### 4. Generate with Local Model
```bash
./generate_template.sh "CSRF vulnerability" /path/to/model.gguf
```

### 5. Interactive Mode
```bash
./interactive_rag.sh
```

## How RAG Works

1. **Index Building**: Creates vector embeddings of all your nuclei templates
2. **Search**: Finds most relevant existing templates for your query
3. **Context Building**: Formats relevant templates as examples
4. **Generation**: Uses LLM with examples to create new template

## Search Types

- **Semantic Search**: Understanding meaning ("injection" finds SQL injection)
- **Keyword Search**: Exact word matching
- **Hybrid Search**: Combines both approaches

## Example Workflow

```python
# Load RAG system
rag = NucleiRAGSystem("~/nuclei-templates")
rag.load_index("indices/nuclei_rag_index.pkl")

# Search for relevant templates
results = rag.hybrid_search("WordPress plugin upload vulnerability", 3)

# Generate new template using examples
llm_function = create_llm_function("/path/to/model.gguf")
new_template = rag.generate_template_with_rag(
    "File upload in WordPress contact plugin", 
    llm_function
)
```

## Benefits vs Fine-tuning

| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| **Setup Time** | Minutes | Hours/Days |
| **Cost** | Free | $50-500 |
| **Updates** | Automatic | Requires retraining |
| **Transparency** | Shows sources | Black box |
| **Quality** | High (with good examples) | High (with good training) |

## Configuration

Edit `rag_config.json` to customize:
- Template paths
- Search weights
- Model settings
- Generation parameters
EOF

echo "✅ Documentation created"

# =============================================================================
# PART 7: Summary
# =============================================================================

echo ""
echo "=== Setup Complete! ==="
echo "✅ Dependencies installed"
echo "✅ Directory structure created"  
echo "✅ Example scripts ready"
echo "✅ Configuration files created"
echo "✅ Wrapper scripts available"
echo "✅ Documentation generated"

echo ""
echo "=== Next Steps ==="
echo "1. Ensure nuclei-templates are at ~/nuclei-templates"
echo "2. Build index: ./build_index.sh"
echo "3. Test search: ./search_templates.sh 'SQL injection'"
echo "4. Try interactive: ./interactive_rag.sh"

echo ""
echo "=== Files Created ==="
echo "📄 nuclei_rag_system.py       - Main RAG system"
echo "📄 requirements_rag.txt       - Python dependencies"  
echo "📄 quick_start_rag.py         - Quick start example"
echo "📄 test_search.py             - Search testing"
echo "📄 rag_config.json            - Configuration"
echo "📄 RAG_USAGE.md               - Usage documentation"
echo "📄 build_index.sh             - Build search index"
echo "📄 interactive_rag.sh         - Interactive mode"
echo "📄 search_templates.sh        - Search only"
echo "📄 generate_template.sh       - Generate templates"

echo ""
echo "🚀 Ready to use RAG for nuclei template generation!"
