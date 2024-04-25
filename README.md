# End-to-End MLOps with Azure ML and Azure DevOps

This repository contains code for building an Azure infrastructure for end-to-end mlops (CI/CD).

## Project Organization
This project is organized into the following directories: `.github/workflows`, `data-science`, `data` `infrastructure` & `mlops/azureml`
- `github/workflows`: contains github actions for automating model training and endpoint deployment
- `data-science`: contains environment dependencies, experimentation notebook and scripts for AzureML components
- `data`: contains data used for machine learning operations
- `infrastructure`: contains code for building MLOps (CI/CD) infrastructure
- `mlops/azureml`: contains train and deployment yaml files


# Getting Started
To replicate this repo you will need 
a. an AzureML subscription
b. Azure DevOps account (with parallelism - to run parallel jobs)

### 1. MLOps Architecture
![classical-ml-architecture](https://github.com/Jeremyugo/AzureMLOps-CI-CD-Passenger-Satisfaction/assets/36512525/d2535fda-47bf-4c35-b1d1-f6d7093c0ea7)

### 2. MLOps Infrastructure build - AzureDevOps
![Infra-build](https://github.com/Jeremyugo/AzureMLOps-CI-CD-Passenger-Satisfaction/assets/36512525/b89493bc-707d-4409-bd11-941632d6bb1d)

### 3. GitHub Actions - Model Training
![github-actions](https://github.com/Jeremyugo/AzureMLOps-CI-CD-Passenger-Satisfaction/assets/36512525/1cd5345d-790b-4a34-b9ba-3e8cf6c47718)

### 4. Machine Learning Job - AzureML
![training](https://github.com/Jeremyugo/AzureMLOps-CI-CD-Passenger-Satisfaction/assets/36512525/a5dcc677-e880-43a1-8e2f-a0f90abcdf1a)

### 5. Registered Model
![model](https://github.com/Jeremyugo/AzureMLOps-CI-CD-Passenger-Satisfaction/assets/36512525/40d75e17-f6a4-4626-9fb8-423faea2a938)
