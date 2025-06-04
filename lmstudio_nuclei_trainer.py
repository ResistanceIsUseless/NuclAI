import os
import json
from pathlib import Path
from typing import List, Dict, Any
import argparse

class LMStudioModelManager:
    def __init__(self, lmstudio_path: str = "~/.lmstudio/models"):
        self.lmstudio_path = Path(lmstudio_path).expanduser().resolve()
        self.available_models = []
        
    def list_available_models(self) -> List[Dict[str, str]]:
        """List all models available in LM Studio"""
        print(f"Scanning LM Studio models in: {self.lmstudio_path}")
        
        if not self.lmstudio_path.exists():
            print(f"LM Studio path not found: {self.lmstudio_path}")
            return []
        
        models = []
        
        # LM Studio typically stores models in publisher/model-name structure
        for publisher_dir in self.lmstudio_path.iterdir():
            if publisher_dir.is_dir():
                for model_dir in publisher_dir.iterdir():
                    if model_dir.is_dir():
                        # Look for GGUF files
                        gguf_files = list(model_dir.glob("*.gguf"))
                        if gguf_files:
                            model_info = {
                                'publisher': publisher_dir.name,
                                'model_name': model_dir.name,
                                'full_path': str(model_dir),
                                'gguf_files': [str(f) for f in gguf_files],
                                'display_name': f"{publisher_dir.name}/{model_dir.name}"
                            }
                            models.append(model_info)
        
        self.available_models = models
        return models
    
    def display_models(self):
        """Display available models in a user-friendly format"""
        models = self.list_available_models()
        
        if not models:
            print("No models found in LM Studio directory!")
            return
        
        print(f"\n=== Available LM Studio Models ===")
        for i, model in enumerate(models, 1):
            print(f"{i:2d}. {model['display_name']}")
            print(f"     Path: {model['full_path']}")
            print(f"     GGUF files: {len(model['gguf_files'])}")
            if model['gguf_files']:
                # Show the main GGUF file (usually the largest or first one)
                main_file = Path(model['gguf_files'][0]).name
                print(f"     Main file: {main_file}")
            print()
    
    def get_model_by_index(self, index: int) -> Dict[str, str]:
        """Get model info by index from the list"""
        if 1 <= index <= len(self.available_models):
            return self.available_models[index - 1]
        return None

class LMStudioNucleiTrainer:
    def __init__(self):
        self.model_manager = LMStudioModelManager()
        
    def setup_llama_cpp(self, model_path: str):
        """Setup llama-cpp-python for training with GGUF models"""
        try:
            from llama_cpp import Llama
            print(f"Loading model from: {model_path}")
            
            # Initialize model for inference/fine-tuning
            self.llama_model = Llama(
                model_path=model_path,
                n_ctx=4096,  # Context length
                n_threads=8,  # Adjust for M4 cores
                n_gpu_layers=-1,  # Use Metal on M4
                verbose=False
            )
            
            print("Model loaded successfully!")
            return True
            
        except ImportError:
            print("llama-cpp-python not installed. Installing...")
            os.system("pip install llama-cpp-python")
            return self.setup_llama_cpp(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def setup_unsloth_for_gguf(self, model_path: str):
        """Alternative: Convert GGUF to HuggingFace format for Unsloth training"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            print(f"Converting GGUF model for training: {model_path}")
            
            # Note: This requires converting GGUF back to HF format
            # You might need to use the original HF model instead
            print("For fine-tuning, you may need the original HuggingFace version of this model")
            print("GGUF is primarily for inference, not training")
            
            return False
            
        except Exception as e:
            print(f"Error setting up model for training: {e}")
            return False
    
    def inference_mode_training(self, model_path: str, dataset_file: str):
        """Use the model for generating training examples (inference-based approach)"""
        if not self.setup_llama_cpp(model_path):
            return False
        
        print("Setting up inference-based training approach...")
        
        # Load training data
        training_examples = []
        with open(dataset_file, 'r') as f:
            for line in f:
                training_examples.append(json.loads(line))
        
        print(f"Loaded {len(training_examples)} training examples")
        
        # Test model with a few examples
        self.test_model_capabilities(training_examples[:3])
        
        return True
    
    def test_model_capabilities(self, examples: List[Dict[str, str]]):
        """Test the model's current capabilities with nuclei templates"""
        print("\n=== Testing Model Capabilities ===")
        
        for i, example in enumerate(examples, 1):
            print(f"\nTest {i}:")
            print(f"Instruction: {example['instruction']}")
            print(f"Expected length: {len(example['response'])} chars")
            
            # Generate response
            prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
            
            try:
                response = self.llama_model(
                    prompt,
                    max_tokens=1000,
                    temperature=0.1,
                    stop=["###", "<|endoftext|>"]
                )
                
                generated = response['choices'][0]['text'].strip()
                print(f"Generated: {generated[:200]}...")
                print(f"Generated length: {len(generated)} chars")
                
            except Exception as e:
                print(f"Error generating: {e}")
    
    def create_fine_tuning_dataset_for_openai_format(self, dataset_file: str, output_file: str = "nuclei_openai_format.jsonl"):
        """Convert dataset to OpenAI fine-tuning format for external training"""
        print(f"Converting dataset to OpenAI format...")
        
        openai_examples = []
        
        with open(dataset_file, 'r') as f:
            for line in f:
                example = json.loads(line)
                
                openai_example = {
                    "messages": [
                        {"role": "system", "content": "You are an expert at creating nuclei templates for security testing."},
                        {"role": "user", "content": example['instruction']},
                        {"role": "assistant", "content": example['response']}
                    ]
                }
                openai_examples.append(openai_example)
        
        with open(output_file, 'w') as f:
            for example in openai_examples:
                json.dump(example, f)
                f.write('\n')
        
        print(f"Saved {len(openai_examples)} examples to {output_file}")
        print("This file can be used with OpenAI's fine-tuning API or similar services")

def main():
    parser = argparse.ArgumentParser(description="Use LM Studio models for nuclei template training")
    parser.add_argument("--list-models", action="store_true", help="List available LM Studio models")
    parser.add_argument("--model-index", type=int, help="Select model by index from list")
    parser.add_argument("--model-path", help="Direct path to GGUF model file")
    parser.add_argument("--dataset", default="nuclei_training_data.jsonl", help="Training dataset file")
    parser.add_argument("--test-only", action="store_true", help="Only test model capabilities")
    parser.add_argument("--create-openai-format", action="store_true", help="Create OpenAI fine-tuning format")
    
    args = parser.parse_args()
    
    trainer = LMStudioNucleiTrainer()
    
    if args.list_models:
        trainer.model_manager.display_models()
        return
    
    if args.create_openai_format:
        trainer.create_fine_tuning_dataset_for_openai_format(args.dataset)
        return
    
    # Determine which model to use
    model_path = None
    
    if args.model_path:
        model_path = args.model_path
    elif args.model_index:
        models = trainer.model_manager.list_available_models()
        model_info = trainer.model_manager.get_model_by_index(args.model_index)
        if model_info and model_info['gguf_files']:
            model_path = model_info['gguf_files'][0]  # Use first GGUF file
        else:
            print(f"Invalid model index: {args.model_index}")
            return
    else:
        # Interactive selection
        trainer.model_manager.display_models()
        try:
            choice = int(input("\nSelect model number: "))
            model_info = trainer.model_manager.get_model_by_index(choice)
            if model_info and model_info['gguf_files']:
                model_path = model_info['gguf_files'][0]
            else:
                print("Invalid selection")
                return
        except (ValueError, KeyboardInterrupt):
            print("Selection cancelled")
            return
    
    if not model_path:
        print("No model selected")
        return
    
    print(f"\nUsing model: {model_path}")
    
    # Check if dataset exists
    if not Path(args.dataset).exists():
        print(f"Dataset file not found: {args.dataset}")
        print("Run nuclei_data_processor.py first to create the dataset")
        return
    
    # Run inference-based training/testing
    success = trainer.inference_mode_training(model_path, args.dataset)
    
    if success:
        print("\n=== Next Steps ===")
        print("1. The model is loaded and tested for nuclei template generation")
        print("2. For actual fine-tuning, consider:")
        print("   - Using the original HuggingFace version of this model")
        print("   - Converting to OpenAI format for external fine-tuning")
        print("   - Using the model as-is for inference-based generation")

if __name__ == "__main__":
    main()
