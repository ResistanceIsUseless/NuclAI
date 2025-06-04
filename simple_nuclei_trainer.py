import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import argparse
from pathlib import Path

class SimpleNucleiTrainer:
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
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model with appropriate settings for M4
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # Use float32 for compatibility
            device_map="auto",  # Automatically use MPS on M4
            trust_remote_code=True
        )
        
        # Ensure model uses pad token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        print(f"Model loaded successfully!")
        return True
    
    def setup_lora_config(self):
        """Setup LoRA for parameter-efficient fine-tuning"""
        # Get just the module names (not full paths)
        target_modules = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module_name = name.split('.')[-1]  # Get just the layer name
                if module_name not in target_modules:
                    target_modules.append(module_name)
        
        print(f"Available module types: {target_modules}")
        
        # For CodeGen, use the attention modules
        if 'qkv_proj' in target_modules:
            selected_modules = ['qkv_proj', 'out_proj']
        elif 'c_attn' in target_modules:
            selected_modules = ['c_attn', 'c_proj']
        else:
            # Fallback to first two linear modules
            selected_modules = target_modules[:2]
        
        print(f"Using LoRA target modules: {selected_modules}")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,  # Reduced LoRA rank for less memory
            lora_alpha=16,  # Reduced alpha
            lora_dropout=0.1,
            target_modules=selected_modules
        )
        
        try:
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        except Exception as e:
            print(f"Error setting up LoRA: {e}")
            raise
        
    def tokenize_function(self, examples):
        """Tokenize the dataset"""
        # Tokenize the text
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,  # We'll pad in the data collator
            max_length=2048,
            return_tensors=None
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
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
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        print(f"Dataset prepared: {len(tokenized_dataset)} examples")
        return tokenized_dataset
    
    def train(self, dataset_file: str, output_dir: str = "./nuclei-fine-tuned"):
        """Fine-tune the model"""
        if not self.model:
            self.load_model_for_training()
        
        # Setup LoRA
        self.setup_lora_config()
        
        # Prepare dataset
        train_dataset = self.prepare_dataset(dataset_file)
        
        # Training arguments optimized for M4 (very conservative)
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,  # Reduced epochs
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,  # Reduced from 4
            warmup_steps=50,  # Reduced warmup
            learning_rate=2e-4,
            logging_steps=5,
            save_steps=100,  # Save more frequently
            eval_strategy="no",
            save_strategy="steps",
            load_best_model_at_end=False,
            report_to=None,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            max_steps=500,  # Conservative max steps
            dataloader_num_workers=0,  # Disable multiprocessing
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            pad_to_multiple_of=8,  # Pad to multiple of 8 for efficiency
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=self.tokenizer,  # Updated parameter name
            data_collator=data_collator,
        )
        
        print("Starting training...")
        print(f"Total training examples: {len(train_dataset)}")
        print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        print(f"Max training steps: {training_args.max_steps}")
        
        # Start training
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
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part
        generated = response.split("[/INST]")[-1].strip()
        
        print(f"\nPrompt: {test_prompt}")
        print(f"Generated Response:\n{generated}")
        
        return generated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--dataset", default="nuclei_training_data.jsonl")
    parser.add_argument("--output", default="./nuclei-fine-tuned")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--test-prompt", default="Create a nuclei template for SQL injection detection")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum training steps")
    
    args = parser.parse_args()
    
    trainer = SimpleNucleiTrainer(args.model)
    
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
        test_prompt = "Create a nuclei template for XSS detection in web forms"
        trainer.test_trained_model(output_path, test_prompt)

if __name__ == "__main__":
    main()
