#!/usr/bin/env python3
"""
Azure LoRA Fine-tuning for Security Models
Configurable fine-tuning script optimized for MacBook Pro M4 Max + LM Studio deployment
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
import time
import subprocess
import yaml

# Azure ML imports
try:
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import (
        Environment, 
        Command,
        Data,
        Model,
        ManagedOnlineEndpoint,
        ManagedOnlineDeployment,
        CodeConfiguration
    )
    from azure.identity import DefaultAzureCredential
    from azure.ai.ml.constants import AssetTypes
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logging.warning("Azure ML SDK not installed. Install with: pip install azure-ai-ml")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for different base models"""
    name: str
    hf_model_id: str
    context_length: int
    recommended_batch_size: int
    recommended_learning_rate: float
    lora_rank: int
    lora_alpha: int
    target_modules: List[str]
    compute_requirements: Dict[str, str]
    m4_max_compatible: bool
    lm_studio_compatible: bool
    description: str

# Model configurations optimized for security fine-tuning and M4 Max compatibility
MODEL_CONFIGS = {
    "phi4-mini-reasoning": ModelConfig(
        name="Phi-4 Mini Reasoning",
        hf_model_id="microsoft/phi-4-mini-reasoning",
        context_length=8192,
        recommended_batch_size=8,
        recommended_learning_rate=2e-4,
        lora_rank=64,
        lora_alpha=128,
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        compute_requirements={"gpu": "Standard_NC12s_v3", "cpu_cores": 12, "memory_gb": 112},
        m4_max_compatible=True,
        lm_studio_compatible=True,
        description="🏆 TOP PICK: Exceptional reasoning capabilities, optimized for step-by-step security analysis. Ultra-fast on M4 Max, perfect for complex security workflows"
    ),
    
    "deepseek-coder-v2-lite": ModelConfig(
        name="DeepSeek Coder V2 Lite Instruct",
        hf_model_id="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        context_length=131072,
        recommended_batch_size=4,
        recommended_learning_rate=1e-4,
        lora_rank=64,
        lora_alpha=128,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        compute_requirements={"gpu": "Standard_NC24ads_A100_v4", "cpu_cores": 24, "memory_gb": 220},
        m4_max_compatible=True,
        lm_studio_compatible=True,
        description="🥇 SECURITY CODING: Superior code understanding and generation. Excellent for vulnerability analysis, exploit development, and security script creation. Large context window for complex analysis"
    ),
    
    "llama3.1-8b-instruct": ModelConfig(
        name="Llama 3.1 8B Instruct",
        hf_model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        context_length=8192,
        recommended_batch_size=4,
        recommended_learning_rate=2e-4,
        lora_rank=64,
        lora_alpha=128,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        compute_requirements={"gpu": "Standard_NC24ads_A100_v4", "cpu_cores": 24, "memory_gb": 220},
        m4_max_compatible=True,
        lm_studio_compatible=True,
        description="Excellent for security reasoning, runs smoothly on M4 Max with 64GB+ RAM. Strong general security analysis"
    ),
    
    "codellama-13b-instruct": ModelConfig(
        name="CodeLlama 13B Instruct", 
        hf_model_id="codellama/CodeLlama-13b-Instruct-hf",
        context_length=16384,
        recommended_batch_size=2,
        recommended_learning_rate=1e-4,
        lora_rank=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        compute_requirements={"gpu": "Standard_NC24ads_A100_v4", "cpu_cores": 24, "memory_gb": 220},
        m4_max_compatible=True,
        lm_studio_compatible=True,
        description="Strong code understanding, good for security script analysis and generation"
    ),
    
    "mistral-7b-instruct": ModelConfig(
        name="Mistral 7B Instruct v0.3",
        hf_model_id="mistralai/Mistral-7B-Instruct-v0.3",
        context_length=32768,
        recommended_batch_size=8,
        recommended_learning_rate=2e-4,
        lora_rank=64,
        lora_alpha=128,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        compute_requirements={"gpu": "Standard_NC12s_v3", "cpu_cores": 12, "memory_gb": 112},
        m4_max_compatible=True,
        lm_studio_compatible=True,
        description="Fast and efficient, excellent reasoning capabilities, perfect for M4 Max"
    ),
    
    "qwen2.5-14b-instruct": ModelConfig(
        name="Qwen2.5 14B Instruct",
        hf_model_id="Qwen/Qwen2.5-14B-Instruct",
        context_length=32768,
        recommended_batch_size=2,
        recommended_learning_rate=1e-4,
        lora_rank=64,
        lora_alpha=128,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        compute_requirements={"gpu": "Standard_NC24ads_A100_v4", "cpu_cores": 24, "memory_gb": 220},
        m4_max_compatible=True,
        lm_studio_compatible=True,
        description="Excellent reasoning and code capabilities, great for complex security analysis"
    ),
    
    "phi3.5-mini-instruct": ModelConfig(
        name="Phi-3.5 Mini Instruct",
        hf_model_id="microsoft/Phi-3.5-mini-instruct",
        context_length=131072,
        recommended_batch_size=16,
        recommended_learning_rate=3e-4,
        lora_rank=32,
        lora_alpha=64,
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        compute_requirements={"gpu": "Standard_NC6s_v3", "cpu_cores": 6, "memory_gb": 112},
        m4_max_compatible=True,
        lm_studio_compatible=True,
        description="Compact but powerful, very fast on M4 Max, good for quick security assessments"
    )
}

class AzureLoRAFineTuner:
    """Azure-based LoRA fine-tuning for security models"""
    
    def __init__(self, subscription_id: str, resource_group: str, workspace_name: str):
        if not AZURE_AVAILABLE:
            raise ImportError("Azure ML SDK not available. Install with: pip install azure-ai-ml")
        
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        
        # Initialize Azure ML client
        credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        
        logger.info(f"Connected to Azure ML workspace: {workspace_name}")
    
    def validate_training_data(self, data_path: str) -> bool:
        """Validate training data format for LoRA fine-tuning"""
        data_file = Path(data_path)
        if not data_file.exists():
            logger.error(f"Training data file not found: {data_path}")
            return False
        
        # Check JSONL format and required fields
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                sample_lines = [f.readline() for _ in range(5) if f.readline()]
            
            for line in sample_lines:
                if line.strip():
                    data = json.loads(line)
                    required_fields = ['instruction', 'input', 'output']
                    if not all(field in data for field in required_fields):
                        logger.error(f"Missing required fields in training data: {required_fields}")
                        return False
            
            logger.info(f"Training data validation passed: {data_path}")
            return True
            
        except Exception as e:
            logger.error(f"Training data validation failed: {e}")
            return False
    
    def upload_training_data(self, data_path: str, data_name: str) -> str:
        """Upload training data to Azure ML"""
        logger.info(f"Uploading training data: {data_path}")
        
        data_asset = Data(
            path=data_path,
            type=AssetTypes.URI_FILE,
            description="Security training data for LoRA fine-tuning",
            name=data_name
        )
        
        data_asset = self.ml_client.data.create_or_update(data_asset)
        logger.info(f"Training data uploaded with version: {data_asset.version}")
        
        return f"{data_name}:{data_asset.version}"
    
    def create_training_environment(self) -> str:
        """Create custom environment for LoRA fine-tuning"""
        environment_name = "security-lora-env"
        
        # Check if environment already exists
        try:
            env = self.ml_client.environments.get(name=environment_name, version="latest")
            logger.info(f"Using existing environment: {environment_name}:{env.version}")
            return f"{environment_name}:{env.version}"
        except:
            logger.info(f"Creating new environment: {environment_name}")
        
        # Create environment with LoRA dependencies
        environment = Environment(
            name=environment_name,
            description="Environment for LoRA fine-tuning of security models",
            conda_file="lora_environment.yml",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu20.04:latest"
        )
        
        # Create conda environment file
        conda_config = {
            'name': 'lora-env',
            'dependencies': [
                'python=3.10',
                'pip',
                {
                    'pip': [
                        'torch>=2.0.0',
                        'transformers>=4.35.0',
                        'peft>=0.6.0',
                        'datasets>=2.14.0',
                        'accelerate>=0.24.0',
                        'bitsandbytes>=0.41.0',
                        'wandb',
                        'tensorboard',
                        'scipy',
                        'scikit-learn',
                        'evaluate',
                        'tqdm'
                    ]
                }
            ]
        }
        
        with open('lora_environment.yml', 'w') as f:
            yaml.dump(conda_config, f)
        
        env = self.ml_client.environments.create_or_update(environment)
        logger.info(f"Environment created: {environment_name}:{env.version}")
        
        return f"{environment_name}:{env.version}"
    
    def create_training_script(self, model_config: ModelConfig, output_dir: str = "./training_scripts"):
        """Create the LoRA fine-tuning training script"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Special handling for different model architectures
        special_imports = ""
        special_model_loading = ""
        
        if "phi-4" in model_config.hf_model_id.lower():
            special_imports = """
# Special imports for Phi-4 reasoning model
from transformers import LlamaTokenizer
import torch.nn.functional as F
"""
            special_model_loading = """
    # Phi-4 specific model loading with reasoning optimization
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,  # Better for reasoning tasks
        device_map="auto",
        load_in_4bit=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
"""
        
        elif "deepseek" in model_config.hf_model_id.lower():
            special_imports = """
# Special imports for DeepSeek Coder model
from transformers.models.llama.tokenization_llama import LlamaTokenizer
"""
            special_model_loading = """
    # DeepSeek Coder specific model loading
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True,
        trust_remote_code=True,
        use_cache=False  # Disable for training
    )
    
    # Enable gradient checkpointing for large context
    model.gradient_checkpointing_enable()
"""
        else:
            special_model_loading = """
    # Standard model loading with 4-bit quantization for efficiency
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True
    )"""

        training_script = f'''#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Security Models
Optimized for {model_config.name}
"""

import os
import json
import torch
import logging
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
import argparse{special_imports}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_training_data(data_path):
    """Load and format training data"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # Format for instruction following with security-specific templates
    formatted_data = []
    for item in data:
        # Enhanced formatting for security reasoning tasks
        reasoning_type = item.get('reasoning_type', 'analysis')
        security_domain = item.get('security_domain', 'general')
        
        if item.get('input'):
            if reasoning_type == 'synthesis':
                text = f"### Security Analysis Task: {{item['instruction']}}\\n### Context: {{item['input']}}\\n### Domain: {security_domain}\\n### Analysis: {{item['output']}}"
            elif reasoning_type == 'exploitation':
                text = f"### Security Assessment: {{item['instruction']}}\\n### Target: {{item['input']}}\\n### Methodology: {{item['output']}}"
            else:
                text = f"### Instruction: {{item['instruction']}}\\n### Input: {{item['input']}}\\n### Response: {{item['output']}}"
        else:
            text = f"### Instruction: {{item['instruction']}}\\n### Response: {{item['output']}}"
        
        formatted_data.append({{"text": text}})
    
    return Dataset.from_list(formatted_data)

def tokenize_function(examples, tokenizer, max_length={model_config.context_length}):
    """Tokenize the training examples"""
    # Tokenize inputs
    model_inputs = tokenizer(
        examples["text"],
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    # Labels are the same as input_ids for causal LM
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    
    return model_inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to training data")
    parser.add_argument("--output_dir", default="./security_lora_model", help="Output directory")
    parser.add_argument("--model_name", default="{model_config.hf_model_id}", help="Base model")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--save_steps", type=int, default=200, help="Save every N steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every N steps")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every N steps")
    
    args = parser.parse_args()
    
    # Load tokenizer and model
    logger.info(f"Loading model: {{args.model_name}}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Special tokenizer settings for specific models
    if "phi-4" in args.model_name.lower():
        # Phi-4 optimizations for reasoning
        tokenizer.padding_side = "left"  # Better for reasoning tasks
    elif "deepseek" in args.model_name.lower():
        # DeepSeek Coder optimizations
        tokenizer.add_eos_token = True
{special_model_loading}
    
    # Configure LoRA with model-specific optimizations
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r={model_config.lora_rank},
        lora_alpha={model_config.lora_alpha},
        lora_dropout=0.1,
        target_modules={model_config.target_modules}
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and tokenize data
    logger.info("Loading training data...")
    dataset = load_training_data(args.data_path)
    
    # Split into train/eval
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    # Tokenize datasets
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    # Model-specific training arguments
    dtype_config = "bf16" if "phi-4" in args.model_name.lower() else "fp16"
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size={model_config.recommended_batch_size},
        per_device_eval_batch_size={model_config.recommended_batch_size},
        gradient_accumulation_steps=2,
        learning_rate={model_config.recommended_learning_rate},
        warmup_steps=100,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_drop_last=True,
        bf16=True if dtype_config == "bf16" else False,
        fp16=True if dtype_config == "fp16" else False,
        report_to="tensorboard",
        run_name="security_lora_finetuning",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        gradient_checkpointing=True
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Start training
    logger.info("Starting LoRA fine-tuning...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {{args.output_dir}}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save model card with security-specific information
    model_card = f"""
# Security-Specialized {model_config.name} LoRA Model

## Model Description
This model is a LoRA fine-tuned version of {model_config.name} specialized for security analysis and reasoning tasks.

## Training Details
- Base Model: {model_config.hf_model_id}
- LoRA Rank: {model_config.lora_rank}
- LoRA Alpha: {model_config.lora_alpha}
- Context Length: {model_config.context_length}
- Specialized for: Security assessment, vulnerability analysis, and reasoning

## Usage
Optimized for security-focused tasks including:
- Vulnerability assessment methodology
- Security tool integration and workflow design  
- Protocol security analysis
- Real-world exploit analysis and mitigation

## Deployment
Compatible with LM Studio on MacBook Pro M4 Max for local security analysis workflows.
"""
    
    with open(os.path.join(args.output_dir, "README.md"), "w") as f:
        f.write(model_card)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
'''
        
        script_path = Path(output_dir) / "lora_training.py"
        with open(script_path, 'w') as f:
            f.write(training_script)
        
        logger.info(f"Training script created: {script_path}")
        return str(script_path)
    
    def submit_training_job(self, 
                           model_config: ModelConfig,
                           data_asset_id: str, 
                           environment_id: str,
                           job_name: str = None) -> str:
        """Submit LoRA fine-tuning job to Azure ML"""
        
        if job_name is None:
            job_name = f"security-lora-{model_config.name.lower().replace(' ', '-')}-{int(time.time())}"
        
        # Create training script
        script_path = self.create_training_script(model_config)
        
        # Configure the training job
        job = Command(
            code="./training_scripts",  # Local directory containing script
            command="python lora_training.py --data_path ${{inputs.training_data}} --output_dir ${{outputs.model_output}} --max_steps 1000",
            environment=environment_id,
            compute=model_config.compute_requirements["gpu"],
            inputs={
                "training_data": {"type": "uri_file", "path": data_asset_id}
            },
            outputs={
                "model_output": {"type": "uri_folder"}
            },
            display_name=job_name,
            description=f"LoRA fine-tuning of {model_config.name} for security reasoning",
            tags={
                "model": model_config.name,
                "task": "security_lora_finetuning",
                "lora_rank": str(model_config.lora_rank),
                "context_length": str(model_config.context_length)
            }
        )
        
        # Submit job
        logger.info(f"Submitting training job: {job_name}")
        submitted_job = self.ml_client.jobs.create_or_update(job)
        
        logger.info(f"Job submitted successfully: {submitted_job.name}")
        logger.info(f"Job status URL: {submitted_job.studio_url}")
        
        return submitted_job.name
    
    def monitor_job(self, job_name: str):
        """Monitor training job progress"""
        logger.info(f"Monitoring job: {job_name}")
        
        job = self.ml_client.jobs.get(job_name)
        logger.info(f"Job status: {job.status}")
        logger.info(f"Studio URL: {job.studio_url}")
        
        return job
    
    def download_trained_model(self, job_name: str, output_path: str = "./trained_models"):
        """Download the trained LoRA model"""
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Downloading trained model from job: {job_name}")
        
        # Download job outputs
        self.ml_client.jobs.download(
            name=job_name,
            output_name="model_output", 
            download_path=str(output_dir)
        )
        
        logger.info(f"Model downloaded to: {output_dir}")
        return output_dir
    
    def create_lm_studio_config(self, model_path: str, model_config: ModelConfig):
        """Create configuration file for LM Studio deployment"""
        config = {
            "model_name": f"Security-{model_config.name}-LoRA",
            "base_model": model_config.hf_model_id,
            "model_path": str(model_path),
            "context_length": model_config.context_length,
            "recommended_settings": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "max_tokens": 2048
            },
            "prompt_template": {
                "instruction": "### Instruction: {instruction}\\n### Input: {input}\\n### Response:",
                "chat": "### Instruction: {instruction}\\n### Response:"
            },
            "deployment_notes": [
                f"Optimized for {model_config.description}",
                f"LoRA Rank: {model_config.lora_rank}, Alpha: {model_config.lora_alpha}",
                "Specialized for security analysis and reasoning",
                "Compatible with M4 Max MacBook Pro"
            ]
        }
        
        config_path = Path(model_path) / "lm_studio_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"LM Studio configuration created: {config_path}")
        return config_path

def list_available_models():
    """List all available model configurations"""
    print("\\n=== Available Models for Security LoRA Fine-tuning ===\\n")
    
    for model_id, config in MODEL_CONFIGS.items():
        compatibility = []
        if config.m4_max_compatible:
            compatibility.append("M4 Max ✓")
        if config.lm_studio_compatible:
            compatibility.append("LM Studio ✓")
        
        print(f"🔧 {model_id}")
        print(f"   Name: {config.name}")
        print(f"   Description: {config.description}")
        print(f"   Context Length: {config.context_length:,}")
        print(f"   Compatibility: {', '.join(compatibility)}")
        print(f"   LoRA Config: Rank={config.lora_rank}, Alpha={config.lora_alpha}")
        print(f"   Azure Compute: {config.compute_requirements['gpu']}")
        print()

def main():
    parser = argparse.ArgumentParser(description="Azure LoRA Fine-tuning for Security Models")
    parser.add_argument("--subscription-id", required=True, help="Azure subscription ID")
    parser.add_argument("--resource-group", required=True, help="Azure resource group")
    parser.add_argument("--workspace-name", required=True, help="Azure ML workspace name")
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), 
                        default="llama3.1-8b-instruct", help="Model to fine-tune")
    parser.add_argument("--training-data", required=True, help="Path to training data JSONL file")
    parser.add_argument("--job-name", help="Custom job name")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--monitor-job", help="Monitor existing job by name")
    parser.add_argument("--download-model", help="Download trained model from job")
    parser.add_argument("--output-path", default="./trained_models", help="Local output path for models")
    
    args = parser.parse_args()
    
    if args.list_models:
        list_available_models()
        return
    
    if not AZURE_AVAILABLE:
        logger.error("Azure ML SDK not installed. Install with: pip install azure-ai-ml")
        return
    
    # Initialize fine-tuner
    fine_tuner = AzureLoRAFineTuner(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name
    )
    
    if args.monitor_job:
        fine_tuner.monitor_job(args.monitor_job)
        return
    
    if args.download_model:
        model_path = fine_tuner.download_trained_model(args.download_model, args.output_path)
        # Create LM Studio config
        model_config = MODEL_CONFIGS[args.model]
        fine_tuner.create_lm_studio_config(model_path, model_config)
        return
    
    # Get model configuration
    model_config = MODEL_CONFIGS[args.model]
    logger.info(f"Selected model: {model_config.name}")
    logger.info(f"M4 Max Compatible: {model_config.m4_max_compatible}")
    logger.info(f"LM Studio Compatible: {model_config.lm_studio_compatible}")
    
    # Validate training data
    if not fine_tuner.validate_training_data(args.training_data):
        logger.error("Training data validation failed")
        return
    
    # Upload training data
    data_name = f"security_training_data_{int(time.time())}"
    data_asset_id = fine_tuner.upload_training_data(args.training_data, data_name)
    
    # Create environment
    environment_id = fine_tuner.create_training_environment()
    
    # Submit training job
    job_name = fine_tuner.submit_training_job(
        model_config=model_config,
        data_asset_id=data_asset_id,
        environment_id=environment_id,
        job_name=args.job_name
    )
    
    logger.info(f"Training job submitted: {job_name}")
    logger.info("Use --monitor-job to check progress")
    logger.info("Use --download-model to get the trained model when complete")

if __name__ == "__main__":
    main()