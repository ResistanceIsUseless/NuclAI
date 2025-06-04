import json
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.tuner import lora
from pathlib import Path
import argparse

class NucleiTrainer:
    def __init__(self, base_model: str = "microsoft/DialoGPT-medium"):
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load base model optimized for M4"""
        print(f"Loading model: {self.base_model}")
        self.model, self.tokenizer = load(self.base_model)
        
    def prepare_dataset(self, jsonl_file: str):
        """Convert JSONL to MLX training format"""
        data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                # Format for instruction following
                text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}<|endoftext|>"
                data.append({"text": text})
        
        return data
    
    def train(self, 
              dataset_file: str = "nuclei_training_data.jsonl",
              output_dir: str = "./nuclei-model",
              num_epochs: int = 3,
              learning_rate: float = 1e-4,
              batch_size: int = 1):
        """Fine-tune model with LoRA"""
        
        if not self.model:
            self.load_model()
            
        # Prepare training data
        train_data = self.prepare_dataset(dataset_file)
        print(f"Training on {len(train_data)} examples")
        
        # Configure LoRA training
        config = {
            "lora_layers": 16,  # Number of layers to apply LoRA
            "lora_rank": 8,     # LoRA rank (lower = less parameters)
            "lora_alpha": 16,   # LoRA scaling parameter
            "lora_dropout": 0.1
        }
        
        # Start training
        print("Starting LoRA training...")
        lora.train(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_data,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            output_dir=output_dir,
            **config
        )
        
        print(f"Training complete! Model saved to {output_dir}")
    
    def test_generation(self, prompt: str, max_tokens: int = 500):
        """Test the trained model"""
        if not self.model:
            print("Model not loaded!")
            return
            
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        
        response = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            temp=0.1  # Low temperature for more consistent code generation
        )
        
        return response

# Alternative Unsloth implementation (more memory efficient)
class UnslothNucleiTrainer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
    
    def setup_unsloth(self, model_name: str = "unsloth/mistral-7b-bnb-4bit"):
        """Setup Unsloth for efficient training"""
        from unsloth import FastLanguageModel
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=None,  # Auto-detect
            load_in_4bit=True,  # Use 4bit quantization for M4
        )
        
        # Setup LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,  # LoRA rank
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=3407,
        )
    
    def train_with_unsloth(self, dataset_file: str):
        """Train using Unsloth's optimized trainer"""
        from datasets import Dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
        
        # Load dataset
        data = []
        with open(dataset_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                data.append({
                    "text": f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}<|endoftext|>"
                })
        
        dataset = Dataset.from_list(data)
        
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            dataset_num_proc=2,
            args=TrainingArguments(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                warmup_steps=10,
                max_steps=100,  # Adjust based on dataset size
                learning_rate=2e-4,
                fp16=not mx.metal.is_available(),  # Use fp16 if not on Metal
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="./nuclei-unsloth-output",
                save_strategy="steps",
                save_steps=50,
            ),
        )
        
        trainer.train()
        
        # Save model
        self.model.save_pretrained("./nuclei-model-unsloth")
        self.tokenizer.save_pretrained("./nuclei-model-unsloth")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=["mlx", "unsloth"], default="mlx")
    parser.add_argument("--dataset", default="nuclei_training_data.jsonl")
    parser.add_argument("--test", action="store_true", help="Test generation")
    args = parser.parse_args()
    
    if args.framework == "mlx":
        trainer = NucleiTrainer()
        if args.test:
            trainer.load_model()
            result = trainer.test_generation("Create a nuclei template for SQL injection detection")
            print("Generated template:", result)
        else:
            trainer.train(args.dataset)
    
    elif args.framework == "unsloth":
        trainer = UnslothNucleiTrainer()
        trainer.setup_unsloth()
        trainer.train_with_unsloth(args.dataset)

if __name__ == "__main__":
    main()
