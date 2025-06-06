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
