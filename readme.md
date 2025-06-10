# Security Model Fine-tuning Suite

A comprehensive toolkit for fine-tuning AI models on security data, supporting both manual Azure AI Foundry fine-tuning and automated Azure ML pipelines.

## 📋 Overview

This suite provides two main approaches for creating security-specialized AI models:

1. **Manual Fine-tuning**: Generate training data → Use Azure AI Foundry UI
2. **Automated Pipeline**: Generate training data → Automated Azure ML training → HuggingFace deployment

## 🚀 Quick Start

### Prerequisites

```bash
# Install required packages
pip install azure-ai-ml azure-identity azure-storage-blob huggingface_hub requests pyyaml

# Setup Azure authentication
az login

# Set HuggingFace token (for automated pipeline)
export HF_TOKEN="your_huggingface_write_token"
```

### Basic Usage

```bash
# 1. Generate security training data
python security_training_data_generator.py --setup --output security_training.jsonl

# 2A. Use with Azure AI Foundry (manual)
# Upload security_training.jsonl in AI Foundry UI

# 2B. OR use automated Azure ML pipeline
python azure_hf_lora_pipeline.py \
  --subscription-id "your-sub-id" \
  --resource-group "your-rg" \
  --workspace-name "your-workspace" \
  --model "microsoft/phi-4-mini-reasoning" \
  --training-data "security_training.jsonl" \
  --hf-repo "yourusername/phi4-security-specialist"
```

---

## 🔧 Part 1: Training Data Generation

### Script: `security_training_data_generator.py`

Generates comprehensive security training data from multiple sources including Nuclei templates, bug bounty reports, and security RFCs.

#### Features

- **Multi-source data**: Nuclei templates, bug bounty reports, RFCs, security documentation
- **Conversational format**: Optimized for chat-based model training
- **Security-focused examples**: Vulnerability analysis, tool creation, methodology guidance
- **Azure Storage support**: Optional upload to Azure Storage Account

#### Usage

```bash
# Basic usage - generate training data locally
python security_training_data_generator.py --setup --output security_training.jsonl

# With Azure Storage upload
python security_training_data_generator.py \
  --setup \
  --output security_training.jsonl \
  --upload-to-storage \
  --storage-connection-string "DefaultEndpointsProtocol=https;AccountName=..."

# Using storage account name + key
python security_training_data_generator.py \
  --setup \
  --upload-to-storage \
  --storage-account-name "yourstorageaccount" \
  --storage-account-key "your_key_here"
```

#### Command Line Options

| Option | Description | Required |
|--------|-------------|----------|
| `--output` | Output file path | No (default: security_training.jsonl) |
| `--setup` | Download/setup data sources | Recommended for first run |
| `--data-path` | Base path for data sources | No (default: ./security_data) |
| `--upload-to-storage` | Upload to Azure Storage | No |
| `--storage-connection-string` | Azure Storage connection string | If uploading |
| `--storage-account-name` | Storage account name | Alternative to connection string |
| `--storage-account-key` | Storage account key | With account name |

#### Output Format

The script generates training data in Azure OpenAI fine-tuning format:

```json
{
  "messages": [
    {"role": "user", "content": "Create a Nuclei template for detecting SQL injection"},
    {"role": "assistant", "content": "I'll create a template for SQL injection detection:\n\n```yaml\nid: sqli-detection\ninfo:\n  name: \"SQL Injection Detection\"\n..."}
  ]
}
```

#### Training Data Sources

1. **Nuclei Templates** (projectdiscovery/nuclei-templates)
   - Vulnerability detection patterns
   - Template creation examples
   - Security testing methodologies

2. **Bug Bounty Reports** (marcotuliocnd/bugbounty-disclosed-reports)
   - Real-world vulnerability examples
   - Testing methodologies
   - Tool integration guidance

3. **Security RFCs**
   - Protocol security analysis
   - Standards compliance
   - Technical security concepts

4. **General Security Knowledge**
   - Scan analysis and next steps
   - Tool integration workflows
   - Cross-tool correlation techniques

---

## 🎯 Part 2A: Manual Fine-tuning with Azure AI Foundry

### Prerequisites

1. **Azure OpenAI resource** with fine-tuning enabled
2. **Training data** generated from Part 1
3. **Access to Azure AI Foundry** (ai.azure.com)

### Step-by-Step Process

#### 1. Access Azure AI Foundry

1. Navigate to [ai.azure.com](https://ai.azure.com)
2. Select your Azure OpenAI resource
3. Go to "Model Management" → "Fine-tuning"
4. Click "Create a fine-tuned model"

#### 2. Configure Fine-tuning Settings

**Basic Settings:**
- **Method**: Reinforcement Learning
- **Base model**: Select your preferred model:
  - `phi-4-mini-reasoning` (recommended for security reasoning)
  - `o4-mini-2025-04-16` (for latest capabilities)
- **Training data**: Upload your `security_training.jsonl` file
- **Validation data**: Use 10-15% of training data
- **Suffix**: `security-specialist`

#### 3. Advanced Configuration

**Hyperparameters (Recommended):**
```yaml
Batch size: 4
Learning rate multiplier: 0.5
Number of epochs: 3
Samples for evaluation: 100
Evaluation interval: 50
Compute multiplier: 1.0
Seed: Random (or 42 for reproducibility)
```

**Response Format (Optional - Flexible):**
```json
{
  "response": {
    "type": "object",
    "properties": {
      "content": {"type": "string"},
      "format": {"type": "string", "enum": ["code", "analysis", "methodology", "explanation", "template"]},
      "additional_context": {"type": "string"}
    },
    "required": ["content"]
  }
}
```

#### 4. Monitor Training

- Training typically takes 1-3 hours
- Monitor progress in the AI Foundry interface
- Check evaluation metrics for overfitting
- Receive email notifications when complete

#### 5. Deploy and Test

1. Deploy the fine-tuned model to an endpoint
2. Test with security-specific prompts
3. Validate model performance on security tasks

### Supported Models

| Model | Best For | Context Length | Memory Requirements |
|-------|----------|----------------|-------------------|
| **phi-4-mini-reasoning** | Complex security reasoning | 8K | Efficient |
| **o4-mini-2025-04-16** | Latest capabilities | 128K | Moderate |

---

## 🤖 Part 2B: Automated Training with Azure ML + HuggingFace

### Script: `azure_hf_lora_pipeline.py`

Automated pipeline that trains models on Azure ML and deploys to HuggingFace Hub for easy local deployment.

### Prerequisites

1. **Azure ML Workspace**
2. **HuggingFace account** with write token
3. **Training data** from Part 1

### Azure ML Workspace Setup

```bash
# Create resource group
az group create --name security-ml-rg --location eastus

# Create ML workspace
az ml workspace create \
  --name security-ml-workspace \
  --resource-group security-ml-rg
```

### Usage

```bash
# Basic training pipeline
python azure_hf_lora_pipeline.py \
  --subscription-id "12345678-1234-1234-1234-123456789abc" \
  --resource-group "security-ml-rg" \
  --workspace-name "security-ml-workspace" \
  --model "microsoft/phi-4-mini-reasoning" \
  --training-data "security_training.jsonl" \
  --hf-repo "yourusername/phi4-security-specialist" \
  --hf-token "hf_your_token_here"

# Monitor existing job
python azure_hf_lora_pipeline.py --monitor-job "security-lora-1234567890"

# Show setup guide
python azure_hf_lora_pipeline.py --setup-guide
```

### Command Line Options

| Option | Description | Required |
|--------|-------------|----------|
| `--subscription-id` | Azure subscription ID | Yes |
| `--resource-group` | Azure resource group | Yes |
| `--workspace-name` | Azure ML workspace name | Yes |
| `--model` | Base model to fine-tune | No (default: phi-4-mini) |
| `--training-data` | Path to training JSONL file | Yes |
| `--hf-repo` | HuggingFace repo (username/model-name) | Yes |
| `--hf-token` | HuggingFace write token | Yes |
| `--monitor-job` | Monitor existing job by name | No |
| `--setup-guide` | Show setup instructions | No |

### Supported Models

| Model | GPU Compute | Cost/Hour | Best For |
|-------|-------------|-----------|----------|
| `microsoft/phi-4-mini-reasoning` | Standard_NC12s_v3 | ~$3.60 | Security reasoning |
| `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | Standard_NC24ads_A100_v4 | ~$3.40 | Code analysis |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | Standard_NC24ads_A100_v4 | ~$3.40 | General security |

### What Happens During Training

1. 🔄 **Upload**: Training data uploaded to Azure ML
2. 🚀 **Provision**: GPU cluster automatically provisioned
3. 🎯 **Train**: LoRA fine-tuning with optimized hyperparameters
4. 📤 **Deploy**: Model automatically pushed to HuggingFace Hub
5. 🛑 **Cleanup**: GPU cluster automatically shut down
6. 💰 **Billing**: Pay only for actual training time (1-3 hours)

### Output

After training completion:
- ✅ **HuggingFace Model**: Available at `https://huggingface.co/yourusername/model-name`
- ✅ **Model Card**: Auto-generated with training details
- ✅ **Version Control**: Git-based versioning on HuggingFace
- ✅ **Ready for Conversion**: Can be converted to GGUF/MLX for local use

---

## 🔄 Part 3: Local Deployment (GGUF/MLX Conversion)

After training (either method), convert your model for local deployment on M4 Max MacBook.

### Download Trained Model

```bash
# Download from HuggingFace (if using automated pipeline)
git clone https://huggingface.co/yourusername/phi4-security-specialist

# Or download from Azure AI Foundry (manual fine-tuning)
# Use Azure AI Foundry interface to download model files
```

### Convert for Local Use

```bash
# Convert to GGUF and MLX formats
python model_conversion_pipeline.py \
  --base-model "microsoft/phi-4-mini-reasoning" \
  --lora-path "./phi4-security-specialist" \
  --model-name "phi4-security-local"
```

### Local Deployment Options

**LM Studio (GUI):**
- Load `.gguf` files from conversion output
- Use Q4_K_M for speed, Q8_0 for quality
- Built-in chat interface

**MLX (Command Line - Fastest on M4 Max):**
```bash
python run_mlx_security_model.py \
  --model ./converted_models/phi4-security-local/mlx/mlx_q4 \
  --prompt "Analyze this Nuclei template for potential false positives"
```

---

## 📊 Comparison: Manual vs Automated

| Aspect | Azure AI Foundry (Manual) | Azure ML Pipeline (Automated) |
|--------|---------------------------|-------------------------------|
| **Setup Complexity** | Low - Use existing OpenAI resource | Medium - Requires ML workspace |
| **Control** | High - Full UI control | Medium - Script-based |
| **Output Format** | Azure OpenAI endpoint | HuggingFace model |
| **Local Deployment** | Manual download + conversion | Automatic HF + conversion |
| **Cost** | Pay per token + training | Pay for GPU compute time |
| **Best For** | Quick experiments, full control | Production workflows, automation |

---

## 🛠️ Troubleshooting

### Common Issues

**Training Data Generation:**
```bash
# If git clone fails
sudo apt-get update && sudo apt-get install git

# If YAML parsing fails
pip install pyyaml

# If requests fail
pip install requests urllib3
```

**Azure Authentication:**
```bash
# Re-authenticate if needed
az login --tenant your-tenant-id

# Check access
az account show
```

**HuggingFace Issues:**
```bash
# Login to HuggingFace CLI
huggingface-cli login

# Test token
python -c "from huggingface_hub import whoami; print(whoami())"
```

### Performance Optimization

**For Training:**
- Use smaller models (phi-4-mini) for faster iteration
- Start with fewer training steps (500) for testing
- Monitor GPU utilization in Azure ML

**For Local Deployment:**
- Q4_K_M quantization for best speed/quality balance
- MLX format for fastest inference on M4 Max
- Use smaller context lengths for better performance

---

## 🎯 Use Cases

### Model Capabilities After Fine-tuning

Your security-specialized model will excel at:

1. **Nuclei Template Creation**
   ```
   User: Create a Nuclei template for detecting Apache Struts RCE
   Model: [Generates complete YAML template with proper detection logic]
   ```

2. **Nmap Script Development**
   ```
   User: Help me write an NSE script for custom service detection
   Model: [Provides Lua script with explanation and usage guidance]
   ```

3. **Security Analysis**
   ```
   User: I found these open ports: 22, 80, 3306. What's my next step?
   Model: [Provides systematic security assessment methodology]
   ```

4. **Tool Integration**
   ```
   User: How do I use Burp Suite to test for SSRF?
   Model: [Explains configuration, methodology, and validation steps]
   ```

5. **CVE Correlation**
   ```
   User: Are there known CVEs for this vulnerability pattern?
   Model: [Provides CVE research methodology and correlation techniques]
   ```

---

## 📝 Example Workflows

### Workflow 1: Quick Experimentation (AI Foundry)

```bash
# 1. Generate data (5 minutes)
python security_training_data_generator.py --setup

# 2. Upload to AI Foundry and configure (5 minutes)
# Use the web interface

# 3. Training (1-3 hours)
# Monitor in AI Foundry interface

# 4. Test via API endpoint
# Use Azure OpenAI SDK
```

### Workflow 2: Production Pipeline (Azure ML)

```bash
# 1. Generate data (5 minutes)
python security_training_data_generator.py --setup

# 2. Submit automated training (2 minutes)
python azure_hf_lora_pipeline.py \
  --subscription-id "your-id" \
  --resource-group "your-rg" \
  --workspace-name "your-workspace" \
  --model "microsoft/phi-4-mini-reasoning" \
  --training-data "security_training.jsonl" \
  --hf-repo "yourusername/phi4-security"

# 3. Monitor training (1-3 hours)
python azure_hf_lora_pipeline.py --monitor-job "job-name"

# 4. Download and convert for local use (10 minutes)
git clone https://huggingface.co/yourusername/phi4-security
python model_conversion_pipeline.py \
  --base-model "microsoft/phi-4-mini-reasoning" \
  --lora-path "./phi4-security"

# 5. Deploy locally in LM Studio or MLX
```

---

## 🤝 Contributing

To improve the training data or add new security domains:

1. **Add new data sources** in `security_training_data_generator.py`
2. **Extend example generation** for new security tools
3. **Improve conversation patterns** for better model responses
4. **Add new model configurations** in the Azure ML pipeline

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## ⚡ Quick Reference

### Essential Commands

```bash
# Generate training data
python security_training_data_generator.py --setup

# Manual fine-tuning: Upload to ai.azure.com

# Automated pipeline
python azure_hf_lora_pipeline.py --subscription-id X --resource-group Y --workspace-name Z --training-data security_training.jsonl --hf-repo username/model

# Monitor training
python azure_hf_lora_pipeline.py --monitor-job job-name

# Convert for local use
python model_conversion_pipeline.py --base-model microsoft/phi-4-mini-reasoning --lora-path ./model
```

### Recommended Model Settings

| Use Case | Model | Batch Size | Learning Rate | Epochs |
|----------|-------|------------|---------------|--------|
| **Quick Testing** | phi-4-mini | 4 | 0.5 | 2 |
| **Production** | phi-4-mini | 4 | 0.5 | 3 |
| **Code Focus** | deepseek-coder | 4 | 0.3 | 3 |
| **General Security** | llama3.1-8b | 4 | 0.5 | 3 |

---

*Happy fine-tuning! 🛡️*