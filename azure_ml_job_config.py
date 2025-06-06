#!/usr/bin/env python3
"""
Azure ML Job Submission Script for Nuclei Template Training
"""

from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import DefaultAzureCredential
import os

class AzureMLJobManager:
    def __init__(self, subscription_id: str, resource_group: str, workspace_name: str):
        self.credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=self.credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
    
    def create_environment(self):
        """Create custom environment for nuclei training"""
        env = Environment(
            name="nuclei-training-env",
            description="Environment for nuclei template training",
            build=BuildContext(
                path=".",
                dockerfile_path="Dockerfile"
            )
        )
        
        # Create environment
        self.ml_client.environments.create_or_update(env)
        return env.name
    
    def create_dockerfile(self):
        """Create Dockerfile for the training environment"""
        dockerfile_content = """
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    wget \\
    curl \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \\
    transformers==4.36.0 \\
    datasets==2.15.0 \\
    peft==0.7.1 \\
    accelerate==0.25.0 \\
    torch==2.1.0 \\
    trl==0.7.4 \\
    wandb \\
    scikit-learn \\
    numpy \\
    pandas \\
    PyYAML \\
    flash-attn --no-build-isolation

# Set working directory
WORKDIR /app

# Copy training scripts
COPY . /app/

# Set environment variables
ENV PYTHONPATH=/app
ENV WANDB_PROJECT=nuclei-training
"""
        
        with open("Dockerfile", "w") as f:
            f.write(dockerfile_content)
        
        print("Dockerfile created")
    
    def submit_training_job(self, data_path: str = "./nuclei_training_data.jsonl"):
        """Submit training job to Azure ML"""
        
        # Create environment first
        env_name = self.create_environment()
        
        # Define the training job
        job = command(
            name="nuclei-template-training",
            display_name="Nuclei Template Model Training",
            description="Fine-tune model for nuclei template generation",
            
            # Environment
            environment=f"{env_name}@latest",
            
            # Compute target
            compute="gpu-cluster",  # We'll create this compute
            
            # Code and data
            code="./",
            inputs={
                "data": Input(type="uri_file", path=data_path),
                "model_name": "deepseek-ai/deepseek-coder-6.7b-instruct",
                "max_steps": 2000,
                "batch_size": 8,
                "learning_rate": 1e-4,
            },
            
            # Outputs
            outputs={
                "model": Output(type="uri_folder", mode="rw_mount"),
                "logs": Output(type="uri_folder", mode="rw_mount")
            },
            
            # Command to run
            command="""
python azure_nuclei_trainer.py \\
    --model ${{inputs.model_name}} \\
    --dataset ${{inputs.data}} \\
    --output ${{outputs.model}} \\
    --max-steps ${{inputs.max_steps}} \\
    --batch-size ${{inputs.batch_size}} \\
    --learning-rate ${{inputs.learning_rate}}
            """,
            
            # Resource requirements
            instance_count=1,
            
            # Tags
            tags={
                "project": "nuclei-templates",
                "model_type": "code-generation",
                "framework": "pytorch"
            }
        )
        
        # Submit job
        submitted_job = self.ml_client.jobs.create_or_update(job)
        
        print(f"Job submitted: {submitted_job.name}")
        print(f"Job URL: {submitted_job.studio_url}")
        
        return submitted_job
    
    def create_compute_cluster(self):
        """Create GPU compute cluster for training"""
        from azure.ai.ml.entities import AmlCompute
        
        compute_config = AmlCompute(
            name="gpu-cluster",
            type="amlcompute",
            size="Standard_NC6s_v3",  # 1x V100 GPU, good for training
            min_instances=0,
            max_instances=1,
            idle_time_before_scale_down=300,  # 5 minutes
            tier="dedicated"
        )
        
        try:
            compute = self.ml_client.compute.create_or_update(compute_config)
            print(f"Compute cluster '{compute.name}' created successfully")
        except Exception as e:
            print(f"Compute cluster creation failed: {e}")
            print("You may need to request quota increase for GPU VMs")
    
    def monitor_job(self, job_name: str):
        """Monitor running job"""
        job = self.ml_client.jobs.get(job_name)
        
        print(f"Job Status: {job.status}")
        print(f"Job URL: {job.studio_url}")
        
        if job.status == "Completed":
            print("Training completed successfully!")
            print(f"Model outputs: {job.outputs}")
        elif job.status == "Failed":
            print("Training failed. Check logs in Azure ML Studio")
        
        return job

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription-id", required=True)
    parser.add_argument("--resource-group", default="nuclei-ml-rg")
    parser.add_argument("--workspace", default="nuclei-training-workspace")
    parser.add_argument("--action", choices=["setup", "submit", "monitor"], required=True)
    parser.add_argument("--job-name", help="Job name for monitoring")
    parser.add_argument("--data-path", default="./nuclei_training_data.jsonl")
    
    args = parser.parse_args()
    
    manager = AzureMLJobManager(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace
    )
    
    if args.action == "setup":
        print("Setting up compute cluster...")
        manager.create_compute_cluster()
        manager.create_dockerfile()
        print("Setup complete!")
        
    elif args.action == "submit":
        print(f"Submitting training job with data: {args.data_path}")
        job = manager.submit_training_job(args.data_path)
        print(f"Job submitted: {job.name}")
        
    elif args.action == "monitor":
        if not args.job_name:
            print("Please provide --job-name for monitoring")
            return
        job = manager.monitor_job(args.job_name)

if __name__ == "__main__":
    main()
