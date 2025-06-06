#!/bin/bash
# Azure ML Workspace Setup Script

# 1. Install Azure CLI (if not already installed)
echo "Installing Azure CLI..."
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# 2. Login to Azure
echo "Logging into Azure..."
az login

# 3. Set your subscription (replace with your subscription ID)
echo "Setting Azure subscription..."
read -p "Enter your Azure Subscription ID: " SUBSCRIPTION_ID
az account set --subscription $SUBSCRIPTION_ID

# 4. Create Resource Group
echo "Creating resource group..."
RESOURCE_GROUP="nuclei-ml-rg"
LOCATION="eastus"  # Change to your preferred region
az group create --name $RESOURCE_GROUP --location $LOCATION

# 5. Create Azure ML Workspace
echo "Creating Azure ML workspace..."
WORKSPACE_NAME="nuclei-training-workspace"
az ml workspace create \
    --name $WORKSPACE_NAME \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION

# 6. Install Azure ML SDK
echo "Installing Azure ML SDK..."
pip install azure-ai-ml azure-identity

echo "Azure ML setup complete!"
echo "Resource Group: $RESOURCE_GROUP"
echo "Workspace: $WORKSPACE_NAME"
echo "Location: $LOCATION"
