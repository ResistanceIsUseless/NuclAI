import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType
import argparse
from pathlib import Path

class NucleiModelTrainer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def load_model_for_training(self):
        """Load HuggingFace model for fine-tuning"""
        print(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate settings for M4
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,  # Use fp16 for memory efficiency
            device_map="auto",  # Automatically use MPS on M4
            trust_remote_code=True
        )
        
        print(f"Model loaded successfully!")
        return True
    
    def setup_lora_config(self):
        """Setup LoRA for parameter-efficient fine-tuning"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # LoRA rank
            lora_alpha=32,  # LoRA scaling parameter
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def prepare_dataset(self, dataset_file: str):
        """Prepare dataset for training"""
        print(f"Loading dataset from: {dataset_file}")
        
        # Load JSONL data
        data = []
        with open(dataset_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                # Format for instruction following
                text = f"<s>[INST] {item['instruction']} [/INST] {item['response']}</s>"
                data.append({"text": text})
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_list(data)
        print(f"Dataset loaded: {len(dataset)} examples")
        
        return dataset
    
    def train(self, dataset_file: str, output_dir: str = "./nuclei-fine-tuned"):
        """Fine-tune the model"""
        if not self.model:
            self.load_model_for_training()
        
        # Setup LoRA
        self.setup_lora_config()
        
        # Prepare dataset
        train_dataset = self.prepare_dataset(dataset_file)
        
        # Training arguments optimized for M4
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,  # Small batch for M4 memory
            gradient_accumulation_steps=4,  # Effective batch size = 4
            warmup_steps=100,
            learning_rate=2e-4,
            logging_steps=10,
            save_steps=100,
            eval_strategy="no",  # Updated parameter name
            save_strategy="steps",
            load_best_model_at_end=False,
            report_to=None,  # Disable wandb
            remove_unused_columns=False,
            dataloader_pin_memory=False,  # Better for M4
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            dataset_text_field="text",
            max_seq_length=2048,
        )
        
        print("Starting training...")
        trainer.train()
        
        # Save the fine-tuned model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Training complete! Model saved to: {output_dir}")
        
        return output_dir
    
    def test_trained_model(self, model_path: str, test_prompt: str):
        """Test the trained model"""
        print(f"Loading trained model from: {model_path}")
        
        # Load the fine-tuned model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Format prompt
        formatted_prompt = f"<s>[INST] {test_prompt} [/INST]"
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part
        generated = response.split("[/INST]")[-1].strip()
        
        print(f"\nPrompt: {test_prompt}")
        print(f"Generated Response:\n{generated}")
        
        return generated

# Common model mappings from GGUF to HuggingFace
MODEL_MAPPINGS = {
    "codellama": "codellama/CodeLlama-7b-Instruct-hf",
    "llama": "meta-llama/Llama-2-7b-chat-hf",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "deepseek": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "phind": "Phind/Phind-CodeLlama-34B-v2",
    "wizardcoder": "WizardLM/WizardCoder-15B-V1.0"
}

def suggest_hf_model():
    """Help user find the right HuggingFace model"""
    print("=== Popular Code Generation Models ===")
    for key, model in MODEL_MAPPINGS.items():
        print(f"{key:12}: {model}")
    
    print("\nTo find your exact model:")
    print("1. Check your LM Studio model name")
    print("2. Search on huggingface.co for the non-GGUF version")
    print("3. Use the model ID (e.g., 'microsoft/DialoGPT-medium')")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--dataset", default="nuclei_training_data.jsonl")
    parser.add_argument("--output", default="./nuclei-fine-tuned")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--test-prompt", default="Create a nuclei template for SQL injection detection")
    parser.add_argument("--suggest-models", action="store_true")
    
    args = parser.parse_args()
    
    if args.suggest_models:
        suggest_hf_model()
        return
    
    trainer = NucleiModelTrainer(args.model)
    
    if args.test_only:
        if Path(args.output).exists():
            trainer.test_trained_model(args.output, args.test_prompt)
        else:
            print(f"Trained model not found at: {args.output}")
    else:
        # Check if dataset exists
        if not Path(args.dataset).exists():
            print(f"Dataset not found: {args.dataset}")
            print("Run nuclei_data_processor.py first!")
            return
        
        # Train the model
        output_path = trainer.train(args.dataset, args.output)
        
        # Test it
        print("\n" + "="*50)
        print("Testing the trained model...")
        trainer.test_trained_model(output_path, args.test_prompt)

if __name__ == "__main__":
    main()
