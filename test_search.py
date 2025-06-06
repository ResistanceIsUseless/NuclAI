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
