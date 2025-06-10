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
| `--setup` | Download/setup data sources (Nuclei, BChecks, Bug Bounty, RFCs) | Recommended for first run |
| `--data-path` | Base path for data sources | No (default: ./security_data) |
| `--upload-to-storage` | Upload to Azure Storage | No |
| `--storage-connection-string` | Azure Storage connection string | If uploading |
| `--storage-account-name` | Storage account name | Alternative to connection string |
| `--storage-account-key` | Storage account key | With account name |

**Enhanced Data Processing:**
- **2,000+ Nuclei templates** processed (increased from 500)
- **500+ Bug bounty reports** analyzed (increased from 100)  
- **800+ BChecks** processed (increased from 200)
- **Comprehensive examples** with 4-5 training samples per source file
- **Real syntax examples** showing actual YAML, BCheck, and exploitation code

#### Output Format

The script generates training data in Azure OpenAI fine-tuning format with **structured, tool-specific examples**:

```json
{
  "messages": [
    {"role": "user", "content": "Analyze this Nuclei template structure and explain how it works:\n\nid: sql-injection-basic\ninfo:\n  name: \"SQL Injection Detection\"\n  severity: high\n  tags: [sqli, injection]\nhttp:\n  - method: GET\n    path:\n      - \"{{BaseURL}}/search?q='\""},
    {"role": "assistant", "content": "This Nuclei template follows the standard structure for vulnerability detection:\n\n**Template Components:**\n\n1. **ID**: sql-injection-basic - Unique identifier\n2. **Info Section**: Contains metadata\n   - Name: SQL Injection Detection\n   - Severity: high\n   - Tags: [sqli, injection]\n\n3. **Detection Logic**: HTTP requests with response pattern matching\n4. **Matchers**: Uses word/regex patterns to identify SQL errors\n\n**How it works:**\n1. Sends HTTP request to target with SQL metacharacter\n2. Analyzes response for database error patterns\n3. Reports positive matches based on error signatures\n4. Classifies finding with appropriate severity level"}
  ]
}
```

**Key Features:**
- **Real syntax examples**: Actual YAML, BCheck syntax, exploitation techniques
- **Structured learning**: Each example teaches specific tool usage patterns  
- **Actionable content**: Focuses on creation and modification rather than explanation
- **Professional depth**: Examples reflect real-world security professional workflows

#### Training Data Sources

1. **Nuclei Templates** (projectdiscovery/nuclei-templates)
   - **Purpose**: Teaching YAML-based vulnerability detection rule creation
   - **Training Focus**: Template structure analysis, matcher logic, variant creation
   - **Examples Generated**: Complete YAML syntax, debugging techniques, optimization strategies
   - **Real-world Application**: Creating custom detection templates for specific applications

2. **Burp BChecks** (PortSwigger/BChecks)
   - **Purpose**: Teaching Burp Suite custom detection logic creation
   - **Training Focus**: BCheck syntax, conditional logic, response analysis
   - **Examples Generated**: Complete BCheck structure, optimization techniques, workflow integration
   - **Real-world Application**: Custom vulnerability detection within Burp Suite workflows

3. **Bug Bounty Reports** (marcotuliocnd/bugbounty-disclosed-reports)
   - **Purpose**: Teaching real-world exploitation methodologies
   - **Training Focus**: "When you see X, try Y" attack patterns, exploitation chains
   - **Examples Generated**: Attack methodologies, tool selection, environmental factors
   - **Real-world Application**: Systematic vulnerability discovery and exploitation techniques

4. **Security RFCs**
   - **Purpose**: Teaching protocol-level security understanding
   - **Training Focus**: Standards compliance, security implications
   - **Examples Generated**: Protocol security analysis, compliance guidance
   - **Real-world Application**: Deep protocol security knowledge for comprehensive assessments

5. **General Security Knowledge**
   - **Purpose**: Teaching comprehensive security methodologies
   - **Training Focus**: Tool integration, assessment workflows, best practices
   - **Examples Generated**: Multi-tool workflows, CI/CD integration, lab setup guides
   - **Real-world Application**: Complete security assessment strategies

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

### Training Data Quality and Scope

**Comprehensive Coverage (5,000+ Examples):**
- **2,000+ Nuclei Templates**: Complete YAML structure, matcher logic, variant creation, debugging techniques
- **800+ BChecks**: Full BCheck syntax, conditional logic, Burp Suite integration workflows
- **500+ Bug Bounty Reports**: Real-world exploitation methodologies, attack chains, tool selection strategies
- **Advanced Methodologies**: Multi-tool integration, CI/CD security, comprehensive assessment frameworks

**Professional Depth:**
- **Structured Rule Creation**: Actual YAML, BCheck syntax with complete examples
- **Exploitation Techniques**: "When you see X, try Y" patterns from successful disclosures  
- **Tool Integration**: Systematic workflows combining multiple security tools
- **Real-world Context**: Examples grounded in professional security practice

**Expected File Size**: 5-15MB (significantly expanded from initial 450KB)
**Training Examples**: 2,000-5,000 high-quality examples
**Content Depth**: 10-20x more detailed than basic security training data

### Model Capabilities After Fine-tuning

Your security-specialized model will excel at:

1. **Structured Detection Rule Creation**
   ```
   User: Create a Nuclei template for detecting Apache Struts RCE
   Model: [Generates complete YAML template with proper detection logic, matchers, and metadata]
   ```

2. **Burp Suite Integration and Automation**
   ```
   User: Write a BCheck for detecting custom authentication bypass
   Model: [Provides complete BCheck syntax with conditional logic and response analysis]
   ```

3. **Exploitation Methodology Guidance**
   ```
   User: I found SQL error messages in parameter testing. What's my next step?
   Model: [Provides systematic exploitation methodology: fingerprinting → injection → escalation]
   ```

4. **Advanced Tool Integration**
   ```
   User: How do I chain Nmap discovery with Nuclei scanning for comprehensive assessment?
   Model: [Explains systematic workflow with specific command sequences and optimization strategies]
   ```

5. **Security Code Analysis**
   ```
   User: Analyze this security script for potential improvements
   Model: [Provides detailed code review with security best practices and optimization suggestions]
   ```

6. **Real-world Attack Simulation**
   ```
   User: Simulate a multi-stage attack against a web application
   Model: [Designs realistic attack chain with tool selection, timing, and evasion techniques]
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