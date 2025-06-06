#!/usr/bin/env python3
"""
Robust Nuclei Template RAG System - M4 Mac Compatible
Handles memory issues and provides fallback options
"""

import json
import yaml
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import argparse
import pickle
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

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

class RobustNucleiRAG:
    def __init__(self, templates_path: str = "~/nuclei-templates", use_simple_search: bool = False):
        self.templates_path = Path(templates_path).expanduser()
        self.templates: List[NucleiTemplate] = []
        self.use_simple_search = use_simple_search
        
        # Initialize search components
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.embeddings_model = None
        self.embeddings = None
        self.simple_index = None  # Fallback search
        
    def load_templates(self):
        """Load and parse all nuclei templates with error handling"""
        print(f"Loading templates from: {self.templates_path}")
        
        if not self.templates_path.exists():
            raise FileNotFoundError(f"Templates path not found: {self.templates_path}")
        
        yaml_files = list(self.templates_path.rglob("*.yaml")) + list(self.templates_path.rglob("*.yml"))
        loaded_count = 0
        error_count = 0
        
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                if not content.strip():
                    continue
                    
                parsed = yaml.safe_load(content)
                
                if parsed and isinstance(parsed, dict) and 'info' in parsed:
                    info = parsed.get('info', {})
                    
                    template = NucleiTemplate(
                        id=parsed.get('id', yaml_file.stem),
                        name=info.get('name', yaml_file.stem),
                        description=info.get('description', ''),
                        severity=info.get('severity', 'info'),
                        category=self._extract_category(yaml_file),
                        file_path=str(yaml_file),
                        raw_content=content,
                        tags=info.get('tags', []) if isinstance(info.get('tags'), list) else [],
                        author=info.get('author', ''),
                        reference=info.get('reference', []) if isinstance(info.get('reference'), list) else []
                    )
                    
                    self.templates.append(template)
                    loaded_count += 1
                    
            except Exception as e:
                error_count += 1
                if error_count <= 5:  # Only print first few errors
                    print(f"Warning: Error loading {yaml_file.name}: {e}")
        
        print(f"✅ Loaded {loaded_count} templates ({error_count} errors)")
        
        if loaded_count == 0:
            raise ValueError("No valid nuclei templates found!")
    
    def _extract_category(self, file_path: Path) -> str:
        """Extract category from file path"""
        try:
            relative_path = file_path.relative_to(self.templates_path)
            return relative_path.parts[0] if len(relative_path.parts) > 1 else 'misc'
        except ValueError:
            return 'misc'
    
    def build_search_index(self):
        """Build search index with fallback options"""
        print("Building search index...")
        
        # Always build TF-IDF (reliable)
        self._build_tfidf_index()
        
        if not self.use_simple_search:
            # Try to build semantic index
            try:
                self._build_semantic_index()
                print("✅ Both TF-IDF and semantic indices built")
            except Exception as e:
                print(f"⚠️  Semantic index failed: {e}")
                print("Falling back to TF-IDF + simple search")
                self.use_simple_search = True
                self._build_simple_index()
        else:
            self._build_simple_index()
            print("✅ TF-IDF and simple search indices built")
    
    def _build_tfidf_index(self):
        """Build TF-IDF index (always works)"""
        print("Building TF-IDF index...")
        
        documents = []
        for template in self.templates:
            # Create searchable document
            doc_parts = [
                template.name,
                template.description,
                template.category,
                template.severity,
                ' '.join(template.tags),
                template.id
            ]
            doc = ' '.join(filter(None, doc_parts)).lower()
            documents.append(doc)
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.8
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        print(f"TF-IDF index built: {self.tfidf_matrix.shape}")
    
    def _build_semantic_index(self):
        """Build semantic search index (may fail on some systems)"""
        print("Building semantic index...")
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Use lighter model for M4 compatibility
            model_name = "all-MiniLM-L6-v2"
            self.embeddings_model = SentenceTransformer(model_name)
            
            # Create texts for embedding
            texts = []
            for template in self.templates:
                text = f"{template.name} {template.description} {template.category}"
                texts.append(text)
            
            # Generate embeddings in smaller batches
            batch_size = 32
            embeddings_list = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.embeddings_model.encode(
                    batch,
                    convert_to_tensor=False,
                    show_progress_bar=False
                )
                embeddings_list.append(batch_embeddings)
            
            self.embeddings = np.vstack(embeddings_list)
            print(f"Semantic embeddings created: {self.embeddings.shape}")
            
        except ImportError:
            raise ImportError("sentence-transformers not available")
        except Exception as e:
            raise RuntimeError(f"Semantic index building failed: {e}")
    
    def _build_simple_index(self):
        """Build simple keyword-based index as fallback"""
        print("Building simple search index...")
        
        self.simple_index = {}
        
        for i, template in enumerate(self.templates):
            # Extract keywords
            keywords = set()
            
            # Add words from name, description, tags
            for text in [template.name, template.description, ' '.join(template.tags)]:
                if text:
                    words = text.lower().replace('-', ' ').replace('_', ' ').split()
                    keywords.update(word.strip('.,()[]{}') for word in words if len(word) > 2)
            
            # Add category and severity
            keywords.add(template.category.lower())
            keywords.add(template.severity.lower())
            
            # Index by keywords
            for keyword in keywords:
                if keyword not in self.simple_index:
                    self.simple_index[keyword] = []
                self.simple_index[keyword].append(i)
        
        print(f"Simple index built with {len(self.simple_index)} keywords")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[NucleiTemplate, float]]:
        """Search templates using available indices"""
        if not self.templates:
            return []
        
        if self.embeddings is not None:
            return self._semantic_search(query, top_k)
        elif self.tfidf_matrix is not None:
            return self._tfidf_search(query, top_k)
        elif self.simple_index:
            return self._simple_search(query, top_k)
        else:
            raise ValueError("No search index available")
    
    def _semantic_search(self, query: str, top_k: int) -> List[Tuple[NucleiTemplate, float]]:
        """Semantic search using embeddings"""
        try:
            query_embedding = self.embeddings_model.encode([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
            
            # Get top results
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum threshold
                    results.append((self.templates[idx], float(similarities[idx])))
            
            return results
        except Exception as e:
            print(f"Semantic search failed: {e}, falling back to TF-IDF")
            return self._tfidf_search(query, top_k)
    
    def _tfidf_search(self, query: str, top_k: int) -> List[Tuple[NucleiTemplate, float]]:
        """TF-IDF based search"""
        query_vector = self.tfidf_vectorizer.transform([query.lower()])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:  # Lower threshold for TF-IDF
                results.append((self.templates[idx], float(similarities[idx])))
        
        return results
    
    def _simple_search(self, query: str, top_k: int) -> List[Tuple[NucleiTemplate, float]]:
        """Simple keyword-based search"""
        query_words = set(query.lower().replace('-', ' ').replace('_', ' ').split())
        template_scores = {}
        
        for word in query_words:
            if word in self.simple_index:
                for template_idx in self.simple_index[word]:
                    template_scores[template_idx] = template_scores.get(template_idx, 0) + 1
        
        # Sort by score
        sorted_results = sorted(template_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for template_idx, score in sorted_results[:top_k]:
            # Normalize score
            normalized_score = score / len(query_words)
            results.append((self.templates[template_idx], normalized_score))
        
        return results
    
    def generate_with_context(self, query: str, llm_function, num_examples: int = 3) -> str:
        """Generate template with context from search results"""
        print(f"Searching for relevant templates...")
        
        relevant_templates = self.search(query, num_examples)
        
        if not relevant_templates:
            print("No relevant templates found, generating basic template")
            return self._generate_basic_template(query)
        
        print(f"Found {len(relevant_templates)} relevant templates")
        for template, score in relevant_templates:
            print(f"- {template.name} (Score: {score:.3f})")
        
        # Build context
        context = self._build_context(relevant_templates)
        prompt = self._create_prompt(query, context)
        
        # Generate with LLM
        try:
            response = llm_function(prompt)
            return response
        except Exception as e:
            print(f"LLM generation failed: {e}")
            return self._generate_basic_template(query)
    
    def _build_context(self, relevant_templates: List[Tuple[NucleiTemplate, float]]) -> str:
        """Build context from relevant templates"""
        context = "Here are relevant nuclei templates for reference:\n\n"
        
        for i, (template, score) in enumerate(relevant_templates, 1):
            context += f"=== Example {i} ===\n"
            context += f"Name: {template.name}\n"
            context += f"Description: {template.description}\n"
            context += f"Severity: {template.severity}\n"
            context += f"Category: {template.category}\n"
            context += "Template:\n"
            context += template.raw_content
            context += "\n" + "="*50 + "\n\n"
        
        return context
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create prompt for LLM"""
        prompt = f"""You are an expert security researcher creating nuclei templates.

{context}

Based on the examples above, create a new nuclei template for: {query}

Requirements:
1. Follow the same YAML structure as the examples
2. Use appropriate severity level
3. Include proper detection logic
4. Make it unique but similar in style

New template:"""
        
        return prompt
    
    def _generate_basic_template(self, query: str) -> str:
        """Generate basic template without examples"""
        return f"""id: basic-{query.lower().replace(' ', '-')[:20]}

info:
  name: {query}
  description: Template for {query}
  severity: medium
  author: rag-system

http:
  - method: GET
    path:
      - "{{{{BaseURL}}}}"
    
    matchers:
      - type: word
        words:
          - "vulnerability"
        condition: and
"""
    
    def save_index(self, path: str):
        """Save index for later use"""
        index_data = {
            'templates': self.templates,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'embeddings': self.embeddings,
            'simple_index': self.simple_index,
            'use_simple_search': self.use_simple_search
        }
        
        with open(path, 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"✅ Index saved to {path}")
    
    def load_index(self, path: str):
        """Load pre-built index"""
        print(f"Loading index from {path}")
        
        with open(path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.templates = index_data['templates']
        self.tfidf_vectorizer = index_data['tfidf_vectorizer']
        self.tfidf_matrix = index_data['tfidf_matrix']
        self.embeddings = index_data.get('embeddings')
        self.simple_index = index_data.get('simple_index')
        self.use_simple_search = index_data.get('use_simple_search', False)
        
        # Reload embeddings model if needed
        if self.embeddings is not None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
            except:
                print("Warning: Could not load embeddings model")
                self.embeddings = None
        
        print(f"✅ Index loaded with {len(self.templates)} templates")

def create_llm_function(model_path: str = None):
    """Create LLM function with error handling"""
    if model_path and Path(model_path).exists():
        try:
            from llama_cpp import Llama
            
            llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=6,
                verbose=False
            )
            
            def generate(prompt: str) -> str:
                response = llm(
                    prompt,
                    max_tokens=1500,
                    temperature=0.1,
                    stop=["===", "Example", "\n\n\n"]
                )
                return response['choices'][0]['text'].strip()
            
            return generate
            
        except Exception as e:
            print(f"Failed to load local model: {e}")
    
    # Mock function for testing
    def mock_generate(prompt: str) -> str:
        return """id: rag-generated-template

info:
  name: RAG Generated Template
  description: Template generated using RAG system
  severity: medium
  author: nuclei-rag

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
    parser = argparse.ArgumentParser(description="Robust Nuclei RAG System")
    parser.add_argument("--templates-path", default="~/nuclei-templates")
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--index-path", default="indices/nuclei_rag_index.pkl")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--model-path", help="Path to local LLM")
    parser.add_argument("--simple-search", action="store_true", help="Use simple search only")
    parser.add_argument("--interactive", action="store_true")
    
    args = parser.parse_args()
    
    # Create output directory
    Path("indices").mkdir(exist_ok=True)
    
    rag = RobustNucleiRAG(args.templates_path, args.simple_search)
    
    try:
        if args.build_index:
            print("Building RAG index...")
            rag.load_templates()
            rag.build_search_index()
            rag.save_index(args.index_path)
            return
        
        # Load existing index
        if Path(args.index_path).exists():
            rag.load_index(args.index_path)
        else:
            print("No index found. Building new index...")
            rag.load_templates()
            rag.build_search_index()
            rag.save_index(args.index_path)
        
        if args.interactive:
            llm_function = create_llm_function(args.model_path)
            
            print("=== Nuclei RAG Interactive Mode ===")
            print("Commands: 'search <query>', 'generate <query>', 'quit'")
            
            while True:
                try:
                    user_input = input("\n> ").strip()
                    
                    if user_input.lower() in ['quit', 'exit']:
                        break
                    
                    if user_input.startswith('search '):
                        query = user_input[7:]
                        results = rag.search(query, 5)
                        print(f"\nFound {len(results)} results:")
                        for template, score in results:
                            print(f"- {template.name} (Score: {score:.3f})")
                            print(f"  {template.description[:100]}...")
                            print(f"  Category: {template.category} | Severity: {template.severity}")
                            print()
                    
                    elif user_input.startswith('generate '):
                        query = user_input[9:]
                        result = rag.generate_with_context(query, llm_function)
                        print("\n" + "="*60)
                        print("Generated Template:")
                        print("="*60)
                        print(result)
                        print("="*60)
                    
                    else:
                        # Default to search
                        results = rag.search(user_input, 3)
                        print(f"\nSearch results:")
                        for template, score in results:
                            print(f"- {template.name} (Score: {score:.3f})")
                
                except KeyboardInterrupt:
                    break
        
        elif args.query:
            if args.search_only:
                results = rag.search(args.query, 5)
                for template, score in results:
                    print(f"{template.name} (Score: {score:.3f})")
            else:
                llm_function = create_llm_function(args.model_path)
                result = rag.generate_with_context(args.query, llm_function)
                print(result)
        
        else:
            print("Use --build-index, --query, or --interactive")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
