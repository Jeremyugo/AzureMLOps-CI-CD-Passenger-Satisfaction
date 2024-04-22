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
To replicate this repo you will need an AzureML subscription and Azure DevOps account with parallelism 

**MLOps Architecture**
![classical-ml-architecture](https://github.com/Jeremyugo/AzureMLOps-CI-CD-Passenger-Satisfaction/assets/36512525/d2535fda-47bf-4c35-b1d1-f6d7093c0ea7)



