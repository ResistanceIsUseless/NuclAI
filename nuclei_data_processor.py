import os
import yaml
import json
from pathlib import Path
from typing import List, Dict, Any
import random

class NucleiDataProcessor:
    def __init__(self, base_path: str = "~/nuclei-templates"):
        self.base_path = Path(base_path).expanduser().resolve()
        self.templates = []
        
        # Verify path exists
        if not self.base_path.exists():
            raise FileNotFoundError(f"Nuclei templates path does not exist: {self.base_path}")
        
        print(f"Processing templates from: {self.base_path}")
        
    def collect_templates(self) -> List[Dict[str, Any]]:
        """Recursively collect all YAML templates from all subdirectories"""
        print(f"Searching for templates in: {self.base_path}")
        
        # Find all YAML files recursively (handles any depth)
        yaml_patterns = ["*.yaml", "*.yml"]
        yaml_files = []
        
        for pattern in yaml_patterns:
            yaml_files.extend(self.base_path.rglob(pattern))
        
        print(f"Found {len(yaml_files)} YAML files to process")
        
        processed_count = 0
        skipped_count = 0
        
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Skip empty files
                if not content.strip():
                    skipped_count += 1
                    continue
                    
                parsed = yaml.safe_load(content)
                
                # Validate it's a nuclei template (has 'info' section)
                if parsed and isinstance(parsed, dict) and 'info' in parsed:
                    self.templates.append({
                        'file_path': str(yaml_file),
                        'relative_path': str(yaml_file.relative_to(self.base_path)),
                        'category': self._extract_category(yaml_file),
                        'raw_content': content,
                        'parsed': parsed,
                        'info': parsed.get('info', {}),
                        'requests': parsed.get('requests', []),
                        'http': parsed.get('http', [])
                    })
                    processed_count += 1
                else:
                    skipped_count += 1
                    
            except Exception as e:
                print(f"Error processing {yaml_file.relative_to(self.base_path)}: {e}")
                skipped_count += 1
                
        print(f"Successfully processed: {processed_count} templates")
        print(f"Skipped: {skipped_count} files")
        return self.templates
    
    def _extract_category(self, file_path: Path) -> str:
        """Extract category from file path relative to base directory"""
        try:
            # Get path relative to base_path
            relative_path = file_path.relative_to(self.base_path)
            parts = relative_path.parts
            
            # Use first directory as category, or 'root' if file is in base directory
            if len(parts) > 1:
                return parts[0]
            else:
                return 'root'
        except ValueError:
            # Fallback if file is outside base_path somehow
            return 'misc'
    
    def create_training_dataset(self) -> List[Dict[str, str]]:
        """Create instruction-following training pairs"""
        training_data = []
        
        for template in self.templates:
            # Template generation tasks
            training_data.extend(self._create_generation_examples(template))
            # Template completion tasks
            training_data.extend(self._create_completion_examples(template))
            # Template explanation tasks
            training_data.extend(self._create_explanation_examples(template))
            
        random.shuffle(training_data)
        return training_data
    
    def _create_generation_examples(self, template: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create template generation examples"""
        examples = []
        info = template['info']
        
        # Basic generation from description
        if 'description' in info:
            examples.append({
                'instruction': f"Create a nuclei template for: {info['description']}",
                'response': template['raw_content']
            })
        
        # Generation from CVE
        if 'classification' in info and 'cve-id' in info['classification']:
            cve_id = info['classification']['cve-id']
            examples.append({
                'instruction': f"Generate a nuclei template for {cve_id}",
                'response': template['raw_content']
            })
        
        # Category-based generation
        examples.append({
            'instruction': f"Write a {template['category']} nuclei template for {info.get('name', 'vulnerability detection')}",
            'response': template['raw_content']
        })
        
        return examples
    
    def _create_completion_examples(self, template: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create template completion examples"""
        examples = []
        lines = template['raw_content'].split('\n')
        
        # Complete from partial template (remove last 30% of lines)
        cutoff = int(len(lines) * 0.7)
        partial_template = '\n'.join(lines[:cutoff])
        
        examples.append({
            'instruction': f"Complete this nuclei template:\n\n{partial_template}",
            'response': template['raw_content']
        })
        
        return examples
    
    def _create_explanation_examples(self, template: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create template explanation examples"""
        examples = []
        
        # Explain what template does
        examples.append({
            'instruction': f"Explain what this nuclei template does:\n\n{template['raw_content']}",
            'response': f"This nuclei template detects {template['info'].get('description', 'vulnerabilities')}. "
                       f"It belongs to the {template['category']} category and "
                       f"uses {len(template.get('requests', template.get('http', [])))} HTTP request(s) to test for the vulnerability."
        })
        
        return examples
    
    def save_dataset(self, dataset: List[Dict[str, str]], output_file: str = "nuclei_training_data.jsonl"):
        """Save dataset in JSONL format"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"Saved {len(dataset)} training examples to {output_file}")

# Usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process nuclei templates for training data")
    parser.add_argument("--templates-path", "-p", 
                       default="~/nuclei-templates",
                       help="Path to nuclei templates directory (default: ~/nuclei-templates)")
    parser.add_argument("--output", "-o",
                       default="nuclei_training_data.jsonl",
                       help="Output file name (default: nuclei_training_data.jsonl)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Just analyze templates without creating training data")
    
    args = parser.parse_args()
    
    try:
        processor = NucleiDataProcessor(args.templates_path)
        templates = processor.collect_templates()
        
        if not templates:
            print("No valid nuclei templates found!")
            exit(1)
        
        if args.dry_run:
            print(f"\n=== DRY RUN - Analysis Only ===")
        else:
            training_data = processor.create_training_dataset()
            processor.save_dataset(training_data, args.output)
        
        print(f"\n=== Dataset Statistics ===")
        print(f"Total templates: {len(templates)}")
        if not args.dry_run:
            print(f"Training examples: {len(training_data)}")
        
        # Show categories and their counts
        categories = {}
        for template in templates:
            cat = template['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"\n=== Template Categories ===")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count} templates")
            
        # Show some example paths
        print(f"\n=== Sample Template Paths ===")
        for i, template in enumerate(templates[:5]):
            print(f"  {template['relative_path']}")
        if len(templates) > 5:
            print(f"  ... and {len(templates) - 5} more")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check the path to your nuclei templates.")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(1)
