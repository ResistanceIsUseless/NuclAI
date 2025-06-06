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
