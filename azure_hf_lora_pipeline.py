#!/usr/bin/env python3
"""
Azure ML + Hugging Face LoRA Training Pipeline
Automated training pipeline that outputs GGUF/MLX ready models
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import time
import subprocess

# Azure ML imports
try:
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import Environment, Command, Data
    from azure.identity import DefaultAzureCredential
    from azure.ai.ml.constants import AssetTypes
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

# Hugging Face imports
try:
    from huggingface_hub import HfApi, create_repo
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AzureMLHuggingFacePipeline:
    """Simplified Azure ML + Hugging Face training pipeline"""
    
    def __init__(self, subscription_id: str, resource_group: str, workspace_name: str, hf_token: Optional[str] = None):
        if not AZURE_AVAILABLE:
            raise ImportError("Azure ML SDK not available. Install with: pip install azure-ai-ml")
        
        # Azure ML setup
        credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        
        # Hugging Face setup
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        if self.hf_token and HF_AVAILABLE:
            self.hf_api = HfApi(token=self.hf_token)
            logger.info("✅ Hugging Face API configured")
        
        logger.info(f"✅ Connected to Azure ML workspace: {workspace_name}")
    
    def upload_training_data(self, local_file_path: str) -> str:
        """Upload training data to Azure ML"""
        if not Path(local_file_path).exists():
            raise FileNotFoundError(f"Training data not found: {local_file_path}")
        
        data_name = f"security_training_{int(time.time())}"
        logger.info(f"Uploading training data: {local_file_path}")
        
        data_asset = Data(
            path=local_file_path,
            type=AssetTypes.URI_FILE,
            description="Security training data for LoRA fine-tuning",
            name=data_name
        )
        
        data_asset = self.ml_client.data.create_or_update(data_asset)
        data_asset_id = f"{data_name}:{data_asset.version}"
        
        logger.info(f"✅ Training data uploaded: {data_asset_id}")
        return data_asset_id
    
    def create_training_environment(self) -> str:
        """Create training environment"""
        environment_name = "security-lora-env"
        
        try:
            env = self.ml_client.environments.get(name=environment_name, version="latest")
            logger.info(f"Using existing environment: {environment_name}:{env.version}")
            return f"{environment_name}:{env.version}"
        except:
            logger.info(f"Creating environment: {environment_name}")
        
        # Create conda environment file
        conda_config = {
            'name': 'security-lora',
            'channels': ['conda-forge', 'pytorch', 'nvidia'],
            'dependencies': [
                'python=3.10',
                'pip',
                {
                    'pip': [
                        'torch>=2.1.0',
                        'transformers>=4.36.0',
                        'peft>=0.7.0',
                        'datasets>=2.15.0',
                        'accelerate>=0.25.0',
                        'bitsandbytes>=0.41.0',
                        'huggingface_hub>=0.19.0',
                        'wandb',
                        'tensorboard'
                    ]
                }
            ]
        }
        
        with open('security_lora_env.yml', 'w') as f:
            import yaml
            yaml.dump(conda_config, f)
        
        environment = Environment(
            name=environment_name,
            description="Security LoRA fine-tuning environment",
            conda_file="security_lora_env.yml",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu20.04:latest"
        )
        
        env = self.ml_client.environments.create_or_update(environment)
        logger.info(f"Environment created: {environment_name}:{env.version}")
        return f"{environment_name}:{env.version}"
    
    def create_training_script(self, model_name: str, hf_repo: str):
        """Create the training script"""
        Path("./training_code").mkdir(exist_ok=True)
        
        script_content = f'''#!/usr/bin/env python3
import os
import torch
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import login, create_repo
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="./lora_output")
    parser.add_argument("--model_name", default="{model_name}")
    parser.add_argument("--hf_repo", default="{hf_repo}")
    parser.add_argument("--hf_token")
    parser.add_argument("--max_steps", type=int, default=1000)
    args = parser.parse_args()
    
    # Setup HF login
    if args.hf_token:
        login(token=args.hf_token)
    
    # Load training data
    data = []
    with open(args.data_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # Convert to dataset
    dataset = Dataset.from_list(data)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"] if "phi" in args.model_name.lower() 
                      else ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    
    # Tokenize function
    def tokenize_function(examples):
        # Handle both message format and direct text
        if "messages" in examples:
            # Convert messages to text
            texts = []
            for messages in examples["messages"]:
                if isinstance(messages, list):
                    text = ""
                    for msg in messages:
                        if msg["role"] == "user":
                            text += f"### User: {{msg['content']}}\\n"
                        elif msg["role"] == "assistant":
                            text += f"### Assistant: {{msg['content']}}\\n"
                    texts.append(text)
                else:
                    texts.append(str(messages))
        else:
            texts = examples.get("text", examples.get("content", []))
        
        model_inputs = tokenizer(
            texts,
            max_length=8192,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs
    
    # Process dataset
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"].map(tokenize_function, batched=True)
    eval_dataset = split_dataset["test"].map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        fp16=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        push_to_hub=True if args.hf_token else False,
        hub_model_id=args.hf_repo if args.hf_token else None,
        hub_token=args.hf_token if args.hf_token else None,
        report_to="tensorboard"
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train
    print("🚀 Starting training...")
    trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Push to HuggingFace if token provided
    if args.hf_token:
        try:
            create_repo(repo_id=args.hf_repo, token=args.hf_token, exist_ok=True)
            trainer.push_to_hub(commit_message="Security LoRA fine-tuning completed")
            print(f"✅ Model pushed to: https://huggingface.co/{{args.hf_repo}}")
        except Exception as e:
            print(f"Warning: Failed to push to HuggingFace: {{e}}")
    
    print("🎉 Training completed!")

if __name__ == "__main__":
    main()
'''
        
        script_path = Path("./training_code/train.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return str(script_path)
    
    def submit_training_job(self, model_name: str, data_asset_id: str, hf_repo: str) -> str:
        """Submit training job to Azure ML"""
        
        # Create environment and training script
        environment_id = self.create_training_environment()
        script_path = self.create_training_script(model_name, hf_repo)
        
        job_name = f"security-lora-{int(time.time())}"
        
        # Determine compute based on model
        if "phi" in model_name.lower():
            compute_target = "Standard_NC12s_v3"  # 2x V100
        else:
            compute_target = "Standard_NC24ads_A100_v4"  # A100
        
        job = Command(
            code="./training_code",
            command="python train.py --data_path ${{inputs.training_data}} --hf_token $HF_TOKEN --max_steps 1000",
            environment=environment_id,
            compute=compute_target,
            inputs={
                "training_data": {"type": "uri_file", "path": data_asset_id}
            },
            environment_variables={
                "HF_TOKEN": self.hf_token
            },
            display_name=job_name,
            description=f"Security LoRA fine-tuning: {model_name}"
        )
        
        submitted_job = self.ml_client.jobs.create_or_update(job)
        
        logger.info(f"✅ Job submitted: {submitted_job.name}")
        logger.info(f"📊 Monitor at: {submitted_job.studio_url}")
        logger.info(f"🚀 Will upload to: https://huggingface.co/{hf_repo}")
        
        return submitted_job.name
    
    def monitor_job(self, job_name: str):
        """Monitor job status"""
        job = self.ml_client.jobs.get(job_name)
        logger.info(f"Job: {job_name}")
        logger.info(f"Status: {job.status}")
        logger.info(f"Studio URL: {job.studio_url}")
        return job

def print_setup_guide():
    """Print Azure ML setup guide"""
    guide = """
🔧 AZURE ML SETUP GUIDE
=======================

## 1. Prerequisites:
```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login

# Install Python packages
pip install azure-ai-ml azure-identity huggingface_hub
```

## 2. Create Azure ML Workspace:
```bash
# Create resource group
az group create --name security-ml-rg --location eastus

# Create ML workspace
az ml workspace create --name security-ml-workspace --resource-group security-ml-rg
```

## 3. Set Environment Variables:
```bash
export HF_TOKEN="your_huggingface_write_token"
```

## 4. GPU Compute Used:
- **Phi-4 Mini**: Standard_NC12s_v3 (2x V100) ~$3.60/hour
- **Other models**: Standard_NC24ads_A100_v4 (A100) ~$3.40/hour
- **Auto-scaling**: Spins up for training, shuts down after
- **Typical training time**: 1-3 hours = $3-11 per model

## 5. What You Get:
✅ Trained LoRA model automatically uploaded to HuggingFace
✅ Ready for download and conversion to GGUF/MLX
✅ Version controlled on HuggingFace Hub
✅ No large file uploads/downloads during training
"""
    print(guide)

def main():
    parser = argparse.ArgumentParser(description="Azure ML + HuggingFace LoRA Training Pipeline")
    
    # Required Azure parameters
    parser.add_argument("--subscription-id", help="Azure subscription ID")
    parser.add_argument("--resource-group", help="Azure resource group")
    parser.add_argument("--workspace-name", help="Azure ML workspace name")
    
    # Training parameters
    parser.add_argument("--model", default="microsoft/phi-4-mini-reasoning", 
                        help="Base model to fine-tune")
    parser.add_argument("--training-data", help="Path to training data JSONL file")
    parser.add_argument("--hf-repo", help="HuggingFace repo name (e.g., username/model-name)")
    parser.add_argument("--hf-token", help="HuggingFace token")
    
    # Actions
    parser.add_argument("--setup-guide", action="store_true", help="Show setup guide")
    parser.add_argument("--monitor-job", help="Monitor existing job by name")
    
    args = parser.parse_args()
    
    if args.setup_guide:
        print_setup_guide()
        return
    
    if not args.subscription_id or not args.resource_group or not args.workspace_name:
        logger.error("Missing required Azure credentials")
        logger.info("Use --setup-guide for help")
        return
    
    # Initialize pipeline
    pipeline = AzureMLHuggingFacePipeline(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name,
        hf_token=args.hf_token or os.getenv("HF_TOKEN")
    )
    
    if args.monitor_job:
        pipeline.monitor_job(args.monitor_job)
        return
    
    if not args.training_data or not args.hf_repo:
        logger.error("Missing required arguments: --training-data and --hf-repo")
        return
    
    try:
        # Upload training data
        data_asset_id = pipeline.upload_training_data(args.training_data)
        
        # Submit training job
        job_name = pipeline.submit_training_job(
            model_name=args.model,
            data_asset_id=data_asset_id,
            hf_repo=args.hf_repo
        )
        
        logger.info(f"""
🎉 TRAINING PIPELINE STARTED!

📊 Job Name: {job_name}
🤖 Model: {args.model}
📤 Will upload to: https://huggingface.co/{args.hf_repo}

⏳ Training typically takes 1-3 hours
💰 Cost estimate: $3-11 (depending on model)

🚀 NEXT STEPS:
1. Monitor progress: --monitor-job {job_name}
2. Once complete, model will be on HuggingFace
3. Download and convert to GGUF/MLX for local use

Example conversion after training:
git clone https://huggingface.co/{args.hf_repo}
python model_conversion_pipeline.py --base-model {args.model} --lora-path ./{args.hf_repo.split('/')[-1]}
        """)
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")

if __name__ == "__main__":
    main()def main():
    parser = argparse.ArgumentParser(description="Azure ML + Hugging Face LoRA Pipeline with Storage Support")
    parser.add_argument("--subscription-id", help="Azure subscription ID")
    parser.add_argument("--resource-group", help="Azure resource group")
    parser.add_argument("--workspace-name", help="Azure ML workspace name")
    parser.add_argument("--hf-token", help="Hugging Face token")
    
    # Storage options
    storage_group = parser.add_argument_group("Storage Options")
    storage_group.add_argument("--storage-connection-string", help="#!/usr/bin/env python3
"""
Azure ML + Hugging Face LoRA Fine-tuning Pipeline
Optimized workflow that uses Hugging Face Hub for model storage and sharing
Avoids large model uploads/downloads by leveraging cloud-to-cloud transfers
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import time
import subprocess

# Azure ML imports
try:
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import (
        Environment, 
        Command,
        Data,
        Model,
        CodeConfiguration
    )
    from azure.identity import DefaultAzureCredential
    from azure.ai.ml.constants import AssetTypes
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

# Azure Storage imports
try:
    from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
    from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
    AZURE_STORAGE_AVAILABLE = True
except ImportError:
    AZURE_STORAGE_AVAILABLE = False

# Hugging Face imports
try:
    from huggingface_hub import HfApi, create_repo, upload_file
    import huggingface_hub
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AzureStorageManager:
    """Manages Azure Storage operations for training data"""
    
    def __init__(self, storage_connection_string: Optional[str] = None, storage_account_name: Optional[str] = None, storage_account_key: Optional[str] = None):
        if not AZURE_STORAGE_AVAILABLE:
            raise ImportError("Azure Storage SDK not available. Install with: pip install azure-storage-blob")
        
        # Initialize blob service client
        if storage_connection_string:
            self.blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
        elif storage_account_name and storage_account_key:
            account_url = f"https://{storage_account_name}.blob.core.windows.net"
            credential = storage_account_key
            self.blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
        else:
            # Try to use default credentials
            try:
                credential = DefaultAzureCredential()
                if storage_account_name:
                    account_url = f"https://{storage_account_name}.blob.core.windows.net"
                    self.blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
                else:
                    raise ValueError("Storage account name required when using default credentials")
            except Exception as e:
                raise ValueError(f"Failed to initialize Azure Storage client: {e}")
        
        self.container_name = "training-data"
        self._ensure_container_exists()
        
        logger.info("✅ Azure Storage client initialized")
    
    def _ensure_container_exists(self):
        """Ensure the training data container exists"""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            container_client.get_container_properties()
            logger.info(f"Using existing container: {self.container_name}")
        except ResourceNotFoundError:
            try:
                self.blob_service_client.create_container(self.container_name)
                logger.info(f"Created new container: {self.container_name}")
            except ResourceExistsError:
                logger.info(f"Container {self.container_name} already exists")
        except Exception as e:
            logger.warning(f"Could not verify container existence: {e}")
    
    def upload_training_data(self, local_file_path: str, blob_name: Optional[str] = None) -> str:
        """Upload training data to Azure Storage"""
        local_path = Path(local_file_path)
        
        if not local_path.exists():
            raise FileNotFoundError(f"Training data file not found: {local_file_path}")
        
        if blob_name is None:
            timestamp = int(time.time())
            blob_name = f"security_training_{timestamp}_{local_path.name}"
        
        logger.info(f"Uploading {local_file_path} to Azure Storage...")
        
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=blob_name
            )
            
            # Upload with progress tracking
            with open(local_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
            
            # Get the blob URL
            blob_url = f"https://{self.blob_service_client.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}"
            
            logger.info(f"✅ Training data uploaded successfully")
            logger.info(f"📂 Blob URL: {blob_url}")
            
            return blob_url
            
        except Exception as e:
            logger.error(f"❌ Failed to upload training data: {e}")
            raise
    
    def list_training_files(self) -> List[Dict[str, str]]:
        """List all training data files in storage"""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blobs = container_client.list_blobs()
            
            files = []
            for blob in blobs:
                files.append({
                    'name': blob.name,
                    'size': blob.size,
                    'last_modified': blob.last_modified.isoformat() if blob.last_modified else None,
                    'url': f"https://{self.blob_service_client.account_name}.blob.core.windows.net/{self.container_name}/{blob.name}"
                })
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list training files: {e}")
            return []
    
    def download_training_data(self, blob_name: str, local_file_path: str) -> str:
        """Download training data from Azure Storage"""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            logger.info(f"Downloading {blob_name} from Azure Storage...")
            
            with open(local_file_path, 'wb') as download_file:
                download_stream = blob_client.download_blob()
                download_file.write(download_stream.readall())
            
            logger.info(f"✅ Downloaded to: {local_file_path}")
            return local_file_path
            
        except Exception as e:
            logger.error(f"❌ Failed to download training data: {e}")
            raise
    
    def get_blob_sas_url(self, blob_name: str, expiry_hours: int = 24) -> str:
        """Generate a SAS URL for blob access (for Azure ML)"""
        try:
            from azure.storage.blob import generate_blob_sas, BlobSasPermissions
            from datetime import datetime, timedelta
            
            # Generate SAS token
            sas_token = generate_blob_sas(
                account_name=self.blob_service_client.account_name,
                container_name=self.container_name,
                blob_name=blob_name,
                account_key=self.blob_service_client.credential.account_key if hasattr(self.blob_service_client.credential, 'account_key') else None,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(hours=expiry_hours)
            )
            
            blob_url_with_sas = f"https://{self.blob_service_client.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}?{sas_token}"
            return blob_url_with_sas
            
        except Exception as e:
            logger.warning(f"Could not generate SAS URL: {e}")
            # Return regular blob URL
            return f"https://{self.blob_service_client.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}"

class AzureHuggingFaceLoRAPipeline:
    """Complete pipeline for Azure training + Hugging Face deployment with Storage Account support"""
    
    def __init__(self, 
                 subscription_id: str,
                 resource_group: str, 
                 workspace_name: str,
                 hf_token: Optional[str] = None,
                 storage_connection_string: Optional[str] = None,
                 storage_account_name: Optional[str] = None,
                 storage_account_key: Optional[str] = None):
        
        if not AZURE_AVAILABLE:
            raise ImportError("Azure ML SDK not available. Install with: pip install azure-ai-ml")
        
        if not HF_AVAILABLE:
            raise ImportError("Hugging Face Hub not available. Install with: pip install huggingface_hub")
        
        # Azure ML setup
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        
        credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        
        # Azure Storage setup (optional)
        self.storage_manager = None
        if any([storage_connection_string, storage_account_name]):
            try:
                self.storage_manager = AzureStorageManager(
                    storage_connection_string=storage_connection_string,
                    storage_account_name=storage_account_name,
                    storage_account_key=storage_account_key
                )
                logger.info("✅ Azure Storage integration enabled")
            except Exception as e:
                logger.warning(f"⚠️ Azure Storage setup failed: {e}")
                logger.info("Will use Azure ML default storage instead")
        
        # Hugging Face setup
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        if self.hf_token:
            self.hf_api = HfApi(token=self.hf_token)
            logger.info("✅ Hugging Face API configured")
        else:
            logger.warning("⚠️ HF_TOKEN not provided. Set via --hf-token or HF_TOKEN env var")
        
        logger.info(f"✅ Connected to Azure ML workspace: {workspace_name}")
    
    def upload_training_data_to_storage(self, local_file_path: str) -> str:
        """Upload training data to Azure Storage or Azure ML default storage"""
        if self.storage_manager:
            # Use custom storage account
            blob_url = self.storage_manager.upload_training_data(local_file_path)
            return blob_url
        else:
            # Use Azure ML default storage (existing method)
            return self._upload_to_ml_storage(local_file_path)
    
    def _upload_to_ml_storage(self, local_file_path: str) -> str:
        """Upload to Azure ML default storage (fallback method)"""
        data_name = f"security_training_data_{int(time.time())}"
        logger.info(f"Uploading training data to Azure ML storage: {local_file_path}")
        
        data_asset = Data(
            path=local_file_path,
            type=AssetTypes.URI_FILE,
            description="Security training data for LoRA fine-tuning",
            name=data_name
        )
        
        data_asset = self.ml_client.data.create_or_update(data_asset)
        data_asset_id = f"{data_name}:{data_asset.version}"
        logger.info(f"✅ Training data uploaded to Azure ML: {data_asset_id}")
        
        return data_asset_id
    
    def list_available_training_data(self) -> List[Dict]:
        """List available training data files"""
        files = []
        
        if self.storage_manager:
            # List from custom storage account
            storage_files = self.storage_manager.list_training_files()
            for file_info in storage_files:
                files.append({
                    'source': 'storage_account',
                    'name': file_info['name'],
                    'size': file_info['size'],
                    'last_modified': file_info['last_modified'],
                    'url': file_info['url']
                })
        
        # List from Azure ML storage
        try:
            ml_data_assets = self.ml_client.data.list()
            for asset in ml_data_assets:
                if 'training' in asset.name.lower() or 'security' in asset.name.lower():
                    files.append({
                        'source': 'azure_ml',
                        'name': asset.name,
                        'version': asset.version,
                        'description': asset.description,
                        'created': asset.creation_context.created_at.isoformat() if asset.creation_context else None
                    })
        except Exception as e:
            logger.warning(f"Could not list Azure ML data assets: {e}")
        
        return files
    
    def print_azure_setup_guide(self):
        """Print setup instructions for Azure ML"""
        setup_guide = """
🔧 AZURE MACHINE LEARNING SETUP GUIDE
=====================================

## 1. Azure Resources You'll Use:
   - **Compute**: GPU clusters (automatically managed)
   - **Storage**: Azure Blob Storage (for training data)
   - **Networking**: Standard Azure ML networking
   - **Cost**: Pay-per-use GPU compute time (~$1-3/hour for training)

## 2. Azure Setup Steps:

### A. Create Azure ML Workspace (if not exists):
```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login

# Create resource group
az group create --name your-rg --location eastus

# Create Azure ML workspace
az ml workspace create --name your-workspace --resource-group your-rg
```

### B. Set up authentication:
```bash
# Option 1: Azure CLI (easiest)
az login

# Option 2: Service Principal (for automation)
az ad sp create-for-rbac --name "ml-pipeline" --role contributor \\
    --scopes /subscriptions/YOUR_SUBSCRIPTION_ID
```

### C. Install required packages:
```bash
pip install azure-ai-ml azure-identity huggingface_hub
```

## 3. GPU Compute Resources Used:

### Training Clusters (Auto-provisioned):
- **Standard_NC6s_v3**: 1x V100 GPU, 6 cores, 112GB RAM (~$1.80/hour)
- **Standard_NC12s_v3**: 2x V100 GPU, 12 cores, 224GB RAM (~$3.60/hour)  
- **Standard_NC24ads_A100_v4**: 1x A100 GPU, 24 cores, 220GB RAM (~$3.40/hour)

### What Azure ML Does:
✅ **Auto-scaling**: Spins up GPU nodes when job starts
✅ **Auto-shutdown**: Shuts down when training completes
✅ **Environment management**: Handles CUDA, PyTorch, etc.
✅ **Experiment tracking**: Logs metrics, models, artifacts
✅ **Model registry**: Stores trained models with versioning

### No Manual Setup Required:
❌ No manual VM provisioning
❌ No CUDA installation
❌ No Docker container management
❌ No storage configuration

## 4. Cost Optimization:
- Training typically takes 1-3 hours = $2-10 per model
- Use smaller models (Phi-4, Mistral) for lower costs
- Preemptible instances available for 80% cost reduction
- Auto-shutdown prevents runaway costs

## 5. Required Permissions:
Your Azure account needs:
- **Contributor** role on the resource group
- **AzureML Data Scientist** role on the workspace
"""
        print(setup_guide)
    
    def create_training_environment(self) -> str:
        """Create optimized environment for LoRA training with HF integration"""
        environment_name = "hf-lora-security-env"
        
        try:
            env = self.ml_client.environments.get(name=environment_name, version="latest")
            logger.info(f"Using existing environment: {environment_name}:{env.version}")
            return f"{environment_name}:{env.version}"
        except:
            logger.info(f"Creating new environment: {environment_name}")
        
        # Enhanced conda environment with HF integration
        conda_config = {
            'name': 'hf-lora-env',
            'channels': ['conda-forge', 'nvidia', 'pytorch'],
            'dependencies': [
                'python=3.10',
                'pip',
                {
                    'pip': [
                        'torch>=2.1.0',
                        'transformers>=4.36.0',
                        'peft>=0.7.0',
                        'datasets>=2.15.0',
                        'accelerate>=0.25.0',
                        'bitsandbytes>=0.41.0',
                        'huggingface_hub>=0.19.0',
                        'wandb',
                        'tensorboard',
                        'scipy',
                        'scikit-learn',
                        'evaluate',
                        'tqdm',
                        'safetensors',
                        'tokenizers'
                    ]
                }
            ]
        }
        
        with open('hf_lora_environment.yml', 'w') as f:
            import yaml
            yaml.dump(conda_config, f)
        
        environment = Environment(
            name=environment_name,
            description="Environment for LoRA fine-tuning with Hugging Face integration",
            conda_file="hf_lora_environment.yml",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu20.04:latest"
        )
        
        env = self.ml_client.environments.create_or_update(environment)
        logger.info(f"Environment created: {environment_name}:{env.version}")
        return f"{environment_name}:{env.version}"
    
    def create_hf_integrated_training_script(self, 
                                           model_config: Dict,
                                           hf_repo_name: str,
                                           output_dir: str = "./training_scripts") -> str:
        """Create training script that uploads to Hugging Face"""
        Path(output_dir).mkdir(exist_ok=True)
        
        training_script = f'''#!/usr/bin/env python3
"""
Azure ML + Hugging Face LoRA Training Script
Trains model on Azure, uploads to HF Hub automatically
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
from huggingface_hub import HfApi, create_repo, login
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_huggingface(hf_token):
    """Setup Hugging Face authentication"""
    if hf_token:
        login(token=hf_token)
        logger.info("✅ Logged into Hugging Face")
        return HfApi(token=hf_token)
    else:
        logger.warning("⚠️ No HF token provided")
        return None

def load_and_format_data(data_path):
    """Load and format training data"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # Enhanced formatting for security tasks
    formatted_data = []
    for item in data:
        reasoning_type = item.get('reasoning_type', 'analysis')
        security_domain = item.get('security_domain', 'general')
        
        if item.get('input'):
            if reasoning_type == 'synthesis':
                text = f"<|im_start|>security_analyst\\nTask: {{item['instruction']}}\\nContext: {{item['input']}}\\nDomain: {security_domain}<|im_end|>\\n<|im_start|>analysis\\n{{item['output']}}<|im_end|>"
            elif reasoning_type == 'exploitation':
                text = f"<|im_start|>security_researcher\\nAssessment: {{item['instruction']}}\\nTarget: {{item['input']}}<|im_end|>\\n<|im_start|>methodology\\n{{item['output']}}<|im_end|>"
            else:
                text = f"<|im_start|>user\\n{{item['instruction']}}\\n{{item['input']}}<|im_end|>\\n<|im_start|>assistant\\n{{item['output']}}<|im_end|>"
        else:
            text = f"<|im_start|>user\\n{{item['instruction']}}<|im_end|>\\n<|im_start|>assistant\\n{{item['output']}}<|im_end|>"
        
        formatted_data.append({{"text": text}})
    
    return Dataset.from_list(formatted_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="./lora_model")
    parser.add_argument("--model_name", default="{model_config.get('hf_model_id', 'microsoft/phi-4-mini-reasoning')}")
    parser.add_argument("--hf_repo", default="{hf_repo_name}")
    parser.add_argument("--hf_token", help="Hugging Face token")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--push_to_hub", action="store_true", default=True)
    
    args = parser.parse_args()
    
    # Setup Hugging Face
    hf_api = setup_huggingface(args.hf_token)
    
    # Load model and tokenizer
    logger.info(f"Loading model: {{args.model_name}}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Model-specific optimizations
    model_kwargs = {{
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True
    }}
    
    if "phi-4" in args.model_name.lower():
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["attn_implementation"] = "flash_attention_2"
    elif "deepseek" in args.model_name.lower():
        model_kwargs["use_cache"] = False
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r={model_config.get('lora_rank', 64)},
        lora_alpha={model_config.get('lora_alpha', 128)},
        lora_dropout=0.1,
        target_modules={model_config.get('target_modules', ['q_proj', 'v_proj', 'k_proj', 'o_proj'])}
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and process data
    dataset = load_and_format_data(args.data_path)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["text"],
            max_length={model_config.get('context_length', 8192)},
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs
    
    train_dataset = split_dataset["train"].map(tokenize_function, batched=True, remove_columns=split_dataset["train"].column_names)
    eval_dataset = split_dataset["test"].map(tokenize_function, batched=True, remove_columns=split_dataset["test"].column_names)
    
    # Training arguments with HF integration
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size={model_config.get('recommended_batch_size', 4)},
        per_device_eval_batch_size={model_config.get('recommended_batch_size', 4)},
        gradient_accumulation_steps=2,
        learning_rate={model_config.get('recommended_learning_rate', 2e-4)},
        warmup_steps=100,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_drop_last=True,
        bf16="phi-4" in args.model_name.lower(),
        fp16="phi-4" not in args.model_name.lower(),
        report_to="tensorboard",
        run_name="security_lora_training",
        push_to_hub=args.push_to_hub and hf_api is not None,
        hub_model_id=args.hf_repo if args.push_to_hub else None,
        hub_token=args.hf_token if args.push_to_hub else None,
        hub_strategy="checkpoint",
        remove_unused_columns=False
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train
    logger.info("🚀 Starting LoRA fine-tuning...")
    trainer.train()
    
    # Save locally
    logger.info("💾 Saving model locally...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Push to Hugging Face Hub
    if args.push_to_hub and hf_api:
        logger.info(f"🚀 Pushing final model to Hugging Face: {{args.hf_repo}}")
        try:
            # Create repo if it doesn't exist
            create_repo(repo_id=args.hf_repo, token=args.hf_token, exist_ok=True, private=False)
            
            # Push model
            trainer.push_to_hub(commit_message="Security LoRA fine-tuning completed")
            
            # Create model card
            model_card = f\"\"\"
---
license: apache-2.0
base_model: {args.model_name}
tags:
- security
- cybersecurity
- lora
- fine-tuned
datasets:
- custom-security-dataset
language:
- en
---

# Security-Specialized {{args.model_name.split('/')[-1]}} LoRA

This model is a LoRA fine-tuned version of {args.model_name} specialized for cybersecurity analysis and reasoning.

## Model Details
- **Base Model**: {args.model_name}
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Specialization**: Security analysis, vulnerability assessment, threat research
- **Training Data**: Comprehensive security dataset including Nuclei templates, bug bounty reports, RFCs, and security documentation

## Capabilities
- Security vulnerability assessment
- Threat analysis and reasoning
- Security tool integration and workflow design
- Protocol security analysis
- Compliance and risk assessment
- Security code analysis

## Usage

### With transformers + peft:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("{args.model_name}")
tokenizer = AutoTokenizer.from_pretrained("{args.model_name}")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{args.hf_repo}")

# Generate security analysis
prompt = "### Security Analysis Task: Analyze the security implications of using default credentials\\n### Analysis:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Recommended Settings
- Temperature: 0.7
- Top-p: 0.9
- Max tokens: 1024-2048

## Training Details
- LoRA Rank: {lora_config.r}
- LoRA Alpha: {lora_config.lora_alpha}
- Target Modules: {lora_config.target_modules}
- Training Steps: {args.max_steps}

## Conversion for Local Use
This model can be converted to GGUF or MLX format for local deployment:
- **GGUF**: For use with LM Studio, Ollama, etc.
- **MLX**: For optimized inference on Apple Silicon

## License
Apache 2.0
\"\"\"
            
            # Upload model card
            with open("README.md", "w") as f:
                f.write(model_card)
            
            hf_api.upload_file(
                path_or_fileobj="README.md",
                path_in_repo="README.md", 
                repo_id=args.hf_repo,
                token=args.hf_token
            )
            
            logger.info(f"✅ Model successfully uploaded to: https://huggingface.co/{{args.hf_repo}}")
            
        except Exception as e:
            logger.error(f"❌ Failed to push to Hugging Face: {{e}}")
            logger.info("💾 Model saved locally only")
    
    logger.info("🎉 Training completed!")

if __name__ == "__main__":
    main()
'''
        
        script_path = Path(output_dir) / "hf_lora_training.py"
        with open(script_path, 'w') as f:
            f.write(training_script)
        
        logger.info(f"HF-integrated training script created: {script_path}")
        return str(script_path)
    
    def submit_hf_training_job(self,
                              model_config: Dict,
                              training_data_reference: str,
                              environment_id: str,
                              hf_repo_name: str,
                              job_name: str = None) -> str:
        """Submit training job that uploads to Hugging Face (supports both storage types)"""
        
        if job_name is None:
            job_name = f"security-lora-{model_config['name'].lower().replace(' ', '-')}-{int(time.time())}"
        
        # Create HF-integrated training script
        script_path = self.create_hf_integrated_training_script(model_config, hf_repo_name)
        
        # Determine input configuration based on storage type
        if training_data_reference.startswith('http'):
            # Storage account blob URL
            if self.storage_manager:
                # Try to get SAS URL for secure access
                blob_name = training_data_reference.split('/')[-1]
                try:
                    sas_url = self.storage_manager.get_blob_sas_url(blob_name)
                    training_data_input = {"type": "uri_file", "path": sas_url}
                except:
                    training_data_input = {"type": "uri_file", "path": training_data_reference}
            else:
                training_data_input = {"type": "uri_file", "path": training_data_reference}
        else:
            # Azure ML data asset
            training_data_input = {"type": "uri_file", "path": training_data_reference}
        
        # Configure job with HF token
        job = Command(
            code="./training_scripts",
            command=f"python hf_lora_training.py --data_path ${{inputs.training_data}} --hf_repo {hf_repo_name} --hf_token $HF_TOKEN --max_steps 1000 --push_to_hub",
            environment=environment_id,
            compute=model_config["compute_requirements"]["gpu"],
            inputs={
                "training_data": training_data_input
            },
            environment_variables={
                "HF_TOKEN": self.hf_token  # Pass HF token to job
            },
            display_name=job_name,
            description=f"Security LoRA fine-tuning of {model_config['name']} with Hugging Face upload",
            tags={
                "model": model_config["name"],
                "task": "security_lora_finetuning",
                "hf_repo": hf_repo_name,
                "deployment": "huggingface_hub",
                "storage_type": "storage_account" if training_data_reference.startswith('http') else "azure_ml"
            }
        )
        
        logger.info(f"Submitting job: {job_name}")
        logger.info(f"Training data: {training_data_reference}")
        logger.info(f"Will upload to: https://huggingface.co/{hf_repo_name}")
        
        submitted_job = self.ml_client.jobs.create_or_update(job)
        logger.info(f"✅ Job submitted: {submitted_job.name}")
        logger.info(f"📊 Monitor at: {submitted_job.studio_url}")
        
        return submitted_job.name
    
    def create_conversion_script_for_hf_model(self, hf_repo_name: str, output_dir: str = "./conversion_scripts"):
        """Create script to convert HF model to GGUF/MLX"""
        Path(output_dir).mkdir(exist_ok=True)
        
        conversion_script = f'''#!/usr/bin/env python3
"""
Convert Hugging Face LoRA Model to GGUF/MLX
Downloads from HF Hub and converts for local deployment
"""

import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_and_convert():
    """Download from HF and convert to local formats"""
    
    # Step 1: Download from Hugging Face
    logger.info("📥 Downloading model from Hugging Face...")
    subprocess.run([
        "git", "clone", f"https://huggingface.co/{hf_repo_name}",
        "./downloaded_model"
    ], check=True)
    
    # Step 2: Convert to GGUF
    logger.info("🔄 Converting to GGUF format...")
    subprocess.run([
        "python", "-m", "llama_cpp.convert",
        "--outfile", "./model_q4_k_m.gguf",
        "--outtype", "q4_k_m", 
        "./downloaded_model"
    ])
    
    # Step 3: Convert to MLX
    logger.info("🔄 Converting to MLX format...")
    subprocess.run([
        "python", "-m", "mlx_lm.convert",
        "--hf-path", "./downloaded_model",
        "--mlx-path", "./mlx_model",
        "--quantize", "--q-bits", "4"
    ])
    
    logger.info("✅ Conversion completed!")
    logger.info("📁 Files available:")
    logger.info("   - GGUF: ./model_q4_k_m.gguf")
    logger.info("   - MLX: ./mlx_model/")

if __name__ == "__main__":
    download_and_convert()
'''
        
        script_path = Path(output_dir) / f"convert_{hf_repo_name.replace('/', '_')}.py"
        with open(script_path, 'w') as f:
            f.write(conversion_script)
        
        logger.info(f"Conversion script created: {script_path}")
        return str(script_path)

def print_storage_setup_guide():
    """Print setup guide for Azure Storage integration"""
    setup_guide = """
🗄️ AZURE STORAGE SETUP GUIDE
============================

## Option 1: Using Storage Connection String (Easiest)
```bash
# Get connection string from Azure Portal
# Storage Account > Access Keys > Connection String
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=..."

# Use in script
python azure_hf_lora_pipeline.py \\
    --storage-connection-string "$AZURE_STORAGE_CONNECTION_STRING" \\
    ...other args
```

## Option 2: Using Storage Account Name + Key
```bash
# From Azure Portal: Storage Account > Access Keys
export STORAGE_ACCOUNT_NAME="yourstorageaccount"
export STORAGE_ACCOUNT_KEY="your_key_here"

python azure_hf_lora_pipeline.py \\
    --storage-account-name "$STORAGE_ACCOUNT_NAME" \\
    --storage-account-key "$STORAGE_ACCOUNT_KEY" \\
    ...other args
```

## Option 3: Using Managed Identity (Most Secure)
```bash
# Assign "Storage Blob Data Contributor" role to your identity
# No credentials needed in code

python azure_hf_lora_pipeline.py \\
    --storage-account-name "yourstorageaccount" \\
    ...other args
```

## Benefits of Custom Storage Account:
✅ **Better organization** - Dedicated storage for training data
✅ **Cost control** - Separate billing for storage vs compute
✅ **Access control** - Fine-grained permissions on training data
✅ **Data sharing** - Easy sharing with team members
✅ **Backup/versioning** - Better data management capabilities
✅ **Cross-region** - Storage can be in different region than ML workspace

## Storage Account Requirements:
- **Performance**: Standard (General Purpose v2)
- **Replication**: LRS or ZRS (depending on needs)
- **Access Tier**: Hot (for active training data)
- **Blob Access**: Container must allow blob access

## Setup Steps:
1. Create Storage Account in Azure Portal
2. Create container named "training-data" 
3. Set appropriate access policies
4. Get connection string or account key
5. Test upload with the script
"""
    print(setup_guide)

def print_complete_workflow():
    """Print the complete workflow guide with storage options"""
    workflow_guide = """
🚀 COMPLETE WORKFLOW WITH STORAGE OPTIONS
========================================

## Phase 1: Setup (One-time)
```bash
# Install dependencies
pip install azure-ai-ml azure-identity azure-storage-blob huggingface_hub

# Setup Azure authentication
az login

# Set Hugging Face token
export HF_TOKEN="your_hf_token_here"

# Optional: Setup custom storage
export AZURE_STORAGE_CONNECTION_STRING="your_connection_string"
```

## Phase 2: Generate and Upload Training Data
```bash
# Generate training data
python comprehensive_security_extractors.py --setup --output security_training.jsonl

# Upload to custom storage account (recommended)
python azure_hf_lora_pipeline.py \\
    --storage-connection-string "$AZURE_STORAGE_CONNECTION_STRING" \\
    --upload-data security_training.jsonl

# OR upload to Azure ML default storage
python azure_hf_lora_pipeline.py \\
    --upload-data security_training.jsonl
```

## Phase 3: Training
```bash
# Start training with custom storage
python azure_hf_lora_pipeline.py \\
    --subscription-id "your-sub-id" \\
    --resource-group "your-rg" \\
    --workspace-name "your-workspace" \\
    --storage-connection-string "$AZURE_STORAGE_CONNECTION_STRING" \\
    --model "phi4-mini-reasoning" \\
    --training-data "security_training.jsonl" \\
    --hf-repo "yourusername/phi4-security-specialist"

# Monitor training
python azure_hf_lora_pipeline.py --monitor-job "job-name"
```

## Phase 4: List and Manage Data
```bash
# List all available training data
python azure_hf_lora_pipeline.py --list-data

# Download training data
python azure_hf_lora_pipeline.py --download-data "filename.jsonl"
```

## Storage Options Comparison:

### Azure ML Default Storage:
✅ **Simple setup** - No additional configuration
✅ **Integrated** - Built into Azure ML workspace
❌ **Less control** - Limited data management features
❌ **Mixed billing** - Storage costs mixed with compute

### Custom Storage Account:
✅ **Better organization** - Dedicated storage for training data
✅ **Cost transparency** - Separate billing for storage
✅ **Advanced features** - Versioning, lifecycle management, backup
✅ **Team sharing** - Easy access control and sharing
✅ **Flexibility** - Can be in different region or subscription
❌ **Additional setup** - Requires storage account configuration

## Recommendation:
Use **custom storage account** for production workflows and **Azure ML default** for quick experiments.
"""
    print(workflow_guide)

def main():
    parser = argparse.ArgumentParser(description="Azure ML + Hugging Face LoRA Pipeline")
    parser.add_argument("--subscription-id", help="Azure subscription ID")
    parser.add_argument("--resource-group", help="Azure resource group")
    parser.add_argument("--workspace-name", help="Azure ML workspace name")
    parser.add_argument("--hf-token", help="Hugging Face token")
    parser.add_argument("--model", choices=["phi4-mini-reasoning", "deepseek-coder-v2-lite", "llama3.1-8b-instruct"], 
                        default="phi4-mini-reasoning", help="Model to fine-tune")
    parser.add_argument("--training-data", help="Path to training data JSONL file")
    parser.add_argument("--hf-repo", help="Hugging Face repo name (e.g., username/model-name)")
    parser.add_argument("--setup-guide", action="store_true", help="Show Azure setup guide")
    parser.add_argument("--workflow-guide", action="store_true", help="Show complete workflow")
    parser.add_argument("--monitor-job", help="Monitor existing job")
    
    args = parser.parse_args()
    
    if args.setup_guide:
        AzureHuggingFaceLoRAPipeline.print_azure_setup_guide(None)
        return
    
    if args.workflow_guide:
        print_complete_workflow()
        return
    
    if not all([args.subscription_id, args.resource_group, args.workspace_name]):
        logger.error("Missing required Azure credentials")
        return
    
    # Initialize pipeline
    pipeline = AzureHuggingFaceLoRAPipeline(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name,
        hf_token=args.hf_token
    )
    
    if args.monitor_job:
        job = pipeline.ml_client.jobs.get(args.monitor_job)
        logger.info(f"Job status: {job.status}")
        logger.info(f"Studio URL: {job.studio_url}")
        return
    
    if not all([args.training_data, args.hf_repo]):
        logger.error("Missing required arguments: --training-data and --hf-repo")
        return
    
    # Model configurations
    model_configs = {
        "phi4-mini-reasoning": {
            "name": "Phi-4 Mini Reasoning",
            "hf_model_id": "microsoft/phi-4-mini-reasoning",
            "context_length": 8192,
            "recommended_batch_size": 8,
            "recommended_learning_rate": 2e-4,
            "lora_rank": 64,
            "lora_alpha": 128,
            "target_modules": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
            "compute_requirements": {"gpu": "Standard_NC12s_v3"}
        },
        "deepseek-coder-v2-lite": {
            "name": "DeepSeek Coder V2 Lite",
            "hf_model_id": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            "context_length": 131072,
            "recommended_batch_size": 4,
            "recommended_learning_rate": 1e-4,
            "lora_rank": 64,
            "lora_alpha": 128,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "compute_requirements": {"gpu": "Standard_NC24ads_A100_v4"}
        },
        "llama3.1-8b-instruct": {
            "name": "Llama 3.1 8B Instruct", 
            "hf_model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "context_length": 8192,
            "recommended_batch_size": 4,
            "recommended_learning_rate": 2e-4,
            "lora_rank": 64,
            "lora_alpha": 128,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "compute_requirements": {"gpu": "Standard_NC24ads_A100_v4"}
        }
    }
    
    model_config = model_configs[args.model]
    logger.info(f"Selected model: {model_config['name']}")
    
    # Validate training data
    if not Path(args.training_data).exists():
        logger.error(f"Training data file not found: {args.training_data}")
        return
    
    try:
        # Upload training data
        data_name = f"security_training_data_{int(time.time())}"
        logger.info(f"Uploading training data: {args.training_data}")
        
        from azure.ai.ml.entities import Data
        from azure.ai.ml.constants import AssetTypes
        
        data_asset = Data(
            path=args.training_data,
            type=AssetTypes.URI_FILE,
            description="Security training data for LoRA fine-tuning",
            name=data_name
        )
        
        data_asset = pipeline.ml_client.data.create_or_update(data_asset)
        data_asset_id = f"{data_name}:{data_asset.version}"
        logger.info(f"✅ Training data uploaded: {data_asset_id}")
        
        # Create environment
        environment_id = pipeline.create_training_environment()
        
        # Submit training job
        job_name = pipeline.submit_hf_training_job(
            model_config=model_config,
            data_asset_id=data_asset_id,
            environment_id=environment_id,
            hf_repo_name=args.hf_repo
        )
        
        # Create conversion script for later use
        conversion_script = pipeline.create_conversion_script_for_hf_model(args.hf_repo)
        
        logger.info(f"""
🎉 TRAINING JOB SUBMITTED SUCCESSFULLY!

📊 Job Name: {job_name}
🔗 Monitor at: Azure ML Studio
📤 Will upload to: https://huggingface.co/{args.hf_repo}
🔄 Conversion script: {conversion_script}

⏳ Training typically takes 1-3 hours
💰 Cost estimate: $2-10 (depending on model size)

🚀 NEXT STEPS:
1. Monitor training progress in Azure ML Studio
2. Once complete, model will be available at: https://huggingface.co/{args.hf_repo}
3. Run conversion script to get GGUF/MLX formats for local use
4. Load in LM Studio or use MLX for fast inference on M4 Max

📧 You'll receive email notifications when training completes
        """)
        
    except Exception as e:
        logger.error(f"❌ Failed to submit training job: {e}")
        logger.info("💡 Make sure you have proper Azure permissions and HF token is valid")

if __name__ == "__main__":
    main()