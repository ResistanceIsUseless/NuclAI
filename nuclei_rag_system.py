#!/usr/bin/env python3
"""
Nuclei Template RAG (Retrieval-Augmented Generation) System
Uses your existing nuclei templates as knowledge base for generation
"""

import json
import yaml
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class NucleiTemplate:
    """Represents a nuclei template with metadata"""
    id: str
    name: str
    description: str
    severity: str
    category: str
    file_path: str
    raw_content: str
    tags: List[str]
    author: str = ""
    reference: List[str] = None

class NucleiRAGSystem:
    def __init__(self, templates_path: str = "~/nuclei-templates"):
        self.templates_path = Path(templates_path).expanduser()
        self.templates: List[NucleiTemplate] = []
        self.embeddings_model = None
        self.vector_index = None
        self.embeddings = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
    def load_templates(self):
        """Load and parse all nuclei templates"""
        print(f"Loading templates from: {self.templates_path}")
        
        yaml_files = list(self.templates_path.rglob("*.yaml")) + list(self.templates_path.rglob("*.yml"))
        
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    parsed = yaml.safe_load(content)
                    
                    if parsed and isinstance(parsed, dict) and 'info' in parsed:
                        info = parsed.get('info', {})
                        
                        template = NucleiTemplate(
                            id=parsed.get('id', yaml_file.stem),
                            name=info.get('name', ''),
                            description=info.get('description', ''),
                            severity=info.get('severity', 'info'),
                            category=self._extract_category(yaml_file),
                            file_path=str(yaml_file),
                            raw_content=content,
                            tags=info.get('tags', []),
                            author=info.get('author', ''),
                            reference=info.get('reference', [])
                        )
                        
                        self.templates.append(template)
                        
            except Exception as e:
                print(f"Error loading {yaml_file}: {e}")
        
        print(f"Loaded {len(self.templates)} nuclei templates")
    
    def _extract_category(self, file_path: Path) -> str:
        """Extract category from file path"""
        try:
            relative_path = file_path.relative_to(self.templates_path)
            return relative_path.parts[0] if len(relative_path.parts) > 1 else 'misc'
        except ValueError:
            return 'misc'
    
    def build_vector_index(self, use_gpu: bool = False):
        """Build vector index for semantic search"""
        print("Building vector index for semantic search...")
        
        # Initialize embedding model
        model_name = "all-MiniLM-L6-v2"  # Fast and good for code/text
        self.embeddings_model = SentenceTransformer(model_name)
        
        # Create searchable text for each template
        template_texts = []
        for template in self.templates:
            # Combine multiple fields for better search
            search_text = f"""
            Name: {template.name}
            Description: {template.description}
            Category: {template.category}
            Severity: {template.severity}
            Tags: {' '.join(template.tags)}
            ID: {template.id}
            """.strip()
            template_texts.append(search_text)
        
        # Generate embeddings
        print("Generating embeddings...")
        self.embeddings = self.embeddings_model.encode(
            template_texts,
            convert_to_tensor=False,
            show_progress_bar=True
        )
        
        # Build FAISS index for fast similarity search
        dimension = self.embeddings.shape[1]
        
        if use_gpu and faiss.get_num_gpus() > 0:
            # GPU index (if available)
            res = faiss.StandardGpuResources()
            self.vector_index = faiss.GpuIndexFlatIP(res, dimension)
        else:
            # CPU index
            self.vector_index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.vector_index.add(self.embeddings)
        
        print(f"Vector index built with {len(self.templates)} templates")
    
    def build_keyword_index(self):
        """Build TF-IDF index for keyword search"""
        print("Building keyword index...")
        
        # Create documents for TF-IDF
        documents = []
        for template in self.templates:
            doc = f"{template.name} {template.description} {template.category} {' '.join(template.tags)}"
            documents.append(doc.lower())
        
        # Build TF-IDF matrix
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        print("Keyword index built")
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[NucleiTemplate, float]]:
        """Search templates using semantic similarity"""
        if not self.embeddings_model or not self.vector_index:
            raise ValueError("Vector index not built. Call build_vector_index() first.")
        
        # Encode query
        query_embedding = self.embeddings_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.vector_index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                results.append((self.templates[idx], float(score)))
        
        return results
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[Tuple[NucleiTemplate, float]]:
        """Search templates using keyword matching"""
        if not self.tfidf_vectorizer or self.tfidf_matrix is None:
            raise ValueError("Keyword index not built. Call build_keyword_index() first.")
        
        # Transform query
        query_vector = self.tfidf_vectorizer.transform([query.lower()])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include relevant results
                results.append((self.templates[idx], float(similarities[idx])))
        
        return results
    
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Tuple[NucleiTemplate, float]]:
        """Combine semantic and keyword search results"""
        # Get results from both methods
        semantic_results = self.semantic_search(query, top_k * 2)
        keyword_results = self.keyword_search(query, top_k * 2)
        
        # Combine and re-rank
        combined_scores = {}
        
        # Add semantic scores (weight: 0.7)
        for template, score in semantic_results:
            combined_scores[template.id] = combined_scores.get(template.id, 0) + (score * 0.7)
        
        # Add keyword scores (weight: 0.3)  
        for template, score in keyword_results:
            combined_scores[template.id] = combined_scores.get(template.id, 0) + (score * 0.3)
        
        # Sort by combined score
        template_lookup = {t.id: t for t in self.templates}
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top results
        results = []
        for template_id, score in sorted_results[:top_k]:
            if template_id in template_lookup:
                results.append((template_lookup[template_id], score))
        
        return results
    
    def generate_template_with_rag(self, query: str, llm_function, num_examples: int = 3) -> str:
        """Generate nuclei template using RAG approach"""
        print(f"Searching for relevant templates for: {query}")
        
        # Find relevant templates
        relevant_templates = self.hybrid_search(query, num_examples)
        
        if not relevant_templates:
            print("No relevant templates found, generating from scratch")
            return self._generate_without_examples(query, llm_function)
        
        # Build context with examples
        context = self._build_context(relevant_templates, query)
        
        # Generate with LLM
        prompt = self._create_rag_prompt(query, context)
        response = llm_function(prompt)
        
        return response
    
    def _build_context(self, relevant_templates: List[Tuple[NucleiTemplate, float]], query: str) -> str:
        """Build context string from relevant templates"""
        context = f"Here are {len(relevant_templates)} similar nuclei templates for reference:\n\n"
        
        for i, (template, score) in enumerate(relevant_templates, 1):
            context += f"=== Example {i} (Similarity: {score:.3f}) ===\n"
            context += f"Name: {template.name}\n"
            context += f"Description: {template.description}\n"
            context += f"Category: {template.category}\n"
            context += f"Severity: {template.severity}\n"
            context += f"Template:\n{template.raw_content}\n\n"
        
        return context
    
    def _create_rag_prompt(self, query: str, context: str) -> str:
        """Create prompt for LLM with RAG context"""
        prompt = f"""You are an expert security researcher creating nuclei templates. Use the provided examples as reference to create a new template.

{context}

Now create a new nuclei template for: {query}

Requirements:
1. Follow the same YAML structure as the examples
2. Use appropriate severity level
3. Include proper matchers and conditions
4. Make the template unique but similar in style
5. Ensure it's a valid nuclei template

Generate the new template:"""
        
        return prompt
    
    def _generate_without_examples(self, query: str, llm_function) -> str:
        """Generate template without examples (fallback)"""
        prompt = f"""Create a nuclei template for: {query}

The template should follow standard nuclei YAML format with:
- id: unique identifier
- info: name, description, severity, author
- http/requests: detection logic
- matchers: conditions for vulnerability detection

Generate a complete nuclei template:"""
        
        return llm_function(prompt)
    
    def save_index(self, index_path: str = "nuclei_rag_index.pkl"):
        """Save the built indices for later use"""
        index_data = {
            'templates': self.templates,
            'embeddings': self.embeddings,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix
        }
        
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        # Save FAISS index separately
        if self.vector_index:
            faiss.write_index(self.vector_index, f"{index_path}.faiss")
        
        print(f"Index saved to {index_path}")
    
    def load_index(self, index_path: str = "nuclei_rag_index.pkl"):
        """Load pre-built indices"""
        print(f"Loading index from {index_path}")
        
        with open(index_path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.templates = index_data['templates']
        self.embeddings = index_data['embeddings']
        self.tfidf_vectorizer = index_data['tfidf_vectorizer']
        self.tfidf_matrix = index_data['tfidf_matrix']
        
        # Load FAISS index
        faiss_path = f"{index_path}.faiss"
        if Path(faiss_path).exists():
            self.vector_index = faiss.read_index(faiss_path)
        
        # Reload embedding model
        self.embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        print(f"Index loaded with {len(self.templates)} templates")

def create_llm_function(model_path: str = None):
    """Create LLM function for generation"""
    if model_path:
        # Use local model (LM Studio/llama.cpp)
        from llama_cpp import Llama
        
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=8,
            verbose=False
        )
        
        def generate(prompt: str) -> str:
            response = llm(
                prompt,
                max_tokens=2000,
                temperature=0.1,
                stop=["Human:", "User:", "\n\n\n"]
            )
            return response['choices'][0]['text'].strip()
        
        return generate
    
    else:
        # Mock function for testing
        def mock_generate(prompt: str) -> str:
            return """id: generated-template

info:
  name: Generated Template
  description: This is a generated template
  severity: medium
  author: rag-system

http:
  - method: GET
    path:
      - "{{BaseURL}}/test"
    matchers:
      - type: word
        words:
          - "test"
"""
        return mock_generate

def main():
    parser = argparse.ArgumentParser(description="Nuclei Template RAG System")
    parser.add_argument("--templates-path", default="~/nuclei-templates")
    parser.add_argument("--build-index", action="store_true", help="Build search index")
    parser.add_argument("--query", help="Query for template generation")
    parser.add_argument("--search-only", action="store_true", help="Only search, don't generate")
    parser.add_argument("--model-path", help="Path to local LLM model")
    parser.add_argument("--index-path", default="nuclei_rag_index.pkl")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    # Initialize RAG system
    rag = NucleiRAGSystem(args.templates_path)
    
    if args.build_index:
        print("Building RAG index...")
        rag.load_templates()
        rag.build_vector_index()
        rag.build_keyword_index()
        rag.save_index(args.index_path)
        print("Index building complete!")
        return
    
    # Load existing index
    if Path(args.index_path).exists():
        rag.load_index(args.index_path)
    else:
        print("No index found. Building new index...")
        rag.load_templates()
        rag.build_vector_index()
        rag.build_keyword_index()
        rag.save_index(args.index_path)
    
    if args.interactive:
        # Interactive mode
        llm_function = create_llm_function(args.model_path)
        
        print("=== Nuclei RAG Interactive Mode ===")
        print("Type 'quit' to exit, 'search <query>' to search only")
        
        while True:
            try:
                user_input = input("\nEnter template request: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    break
                
                if user_input.startswith('search '):
                    query = user_input[7:]
                    results = rag.hybrid_search(query, 5)
                    
                    print(f"\nFound {len(results)} relevant templates:")
                    for template, score in results:
                        print(f"- {template.name} (Score: {score:.3f})")
                        print(f"  Description: {template.description}")
                        print(f"  Category: {template.category}, Severity: {template.severity}")
                        print()
                
                else:
                    print("Generating template with RAG...")
                    result = rag.generate_template_with_rag(user_input, llm_function)
                    print("\nGenerated Template:")
                    print("=" * 60)
                    print(result)
                    print("=" * 60)
                
            except KeyboardInterrupt:
                break
    
    elif args.query:
        if args.search_only:
            results = rag.hybrid_search(args.query, 5)
            print(f"Search results for: {args.query}")
            for template, score in results:
                print(f"- {template.name} (Score: {score:.3f})")
                print(f"  {template.description}")
        else:
            llm_function = create_llm_function(args.model_path)
            result = rag.generate_template_with_rag(args.query, llm_function)
            print(result)
    
    else:
        print("Use --interactive, --query, or --build-index")
        parser.print_help()

if __name__ == "__main__":
    main()
