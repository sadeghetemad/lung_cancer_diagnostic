# 🧠 Multimodal Lung Cancer Survival Prediction Pipeline

An end-to-end **multimodal machine learning system** for lung cancer survival prediction using **clinical, genomic, and CT imaging data**, fully built and orchestrated on **AWS SageMaker**.

---

## 🚀 Overview

This project implements a **production-oriented multimodal ML pipeline** where all stages — from preprocessing to training and deployment — are modularized and integrated with AWS services.

Three data modalities are used:

- 🧬 Genomic Data  
- 🏥 Clinical Data  
- 🩻 CT Imaging Data  

Each modality is processed independently and stored in **SageMaker Feature Store**, enabling scalable feature reuse and multimodal training.

---

## ☁️ AWS-Native Architecture

The entire pipeline is designed to run on AWS:

- **SageMaker Processing Jobs** → heavy CT image processing  
- **Docker containers** → radiomics feature extraction  
- **SageMaker Feature Store** → centralized feature storage  
- **Amazon Athena** → feature querying & joins  
- **SageMaker Training (XGBoost)** → model training  
- **SageMaker Endpoint** → real-time inference  

---

## 🧩 Pipeline Structure

### 🔹 1. Genomic Pipeline
- Reads genomic data from S3  
- Selects relevant genes  
- Cleans and transforms features  
- Stores into Feature Store  

### 🔹 2. Clinical Pipeline
- Cleans structured EHR-like data  
- Handles categorical encoding  
- Removes leakage features  
- Stores into Feature Store  

### 🔹 3. Imaging Pipeline
- DICOM → NIfTI conversion  
- 3D reconstruction  
- Radiomics feature extraction  
- Runs via SageMaker Processing + Docker
- Stores into Feature Store  

### 🔹 4. Training & Deployment
- Query multimodal features (Athena)  
- Apply StandardScaler + PCA  
- Train XGBoost  
- Deploy to SageMaker Endpoint  
- Evaluate model  

---

## 🔄 End-to-End Flow

Raw Data → Preprocessing → Feature Store → Athena Join → Training → PCA → Model → Endpoint → Prediction

---

## ⚠️ Important Note

The model expects **scaled + PCA-transformed features** at inference time.

---

## 🧪 Project Structure

notebooks/
- preprocess notebooks

src/
- preprocessing scripts
- image_processing/
- training script

---

## 🔥 Key Highlights

- True multimodal ML (clinical + genomic + imaging)
- Fully AWS-native (no fake local pipelines)
- Scalable CT processing with Docker + SageMaker
- Radiomics feature engineering from 3D volumes
- Feature Store-driven architecture
- Production-ready design (training + endpoint)

---

## 🎯 Goal

Build a scalable multimodal AI system for lung cancer survival prediction.

