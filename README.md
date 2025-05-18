# Student Performance End-to-End Machine Learning Project

## Overview

This repository contains an end-to-end machine learning pipeline for analyzing and predicting student performance. The project is built using modular components with a focus on maintainability, scalability, and production-readiness.

This implementation follows best practices for real-world ML system design including proper logging, exception handling, configuration management, and reproducibility.

---
## 1. Objectives Covered Till Now

### ✅ Exploratory Data Analysis (EDA)

Performed a detailed EDA in `notebook/data_analysis.ipynb`, including:

- Checking data structure, types, nulls, and distribution
- Visualizations for feature correlations and outliers
- Identifying preprocessing steps for the next pipeline stage

---
### ✅ Logging

- Custom logger created using Python's `logging` module
- Logs include:
  - Successful steps
  - Data flow checkpoints
  - Helpful for debugging or pipeline monitoring

> 📂 `src/logger.py`

---

### ✅ Custom Exception Handling

- Centralized error management using a `CustomException` class
- Catches and formats exceptions with traceback and error context
- Used across the ingestion module for better fault tolerance

> 📂 `src/exception.py`

---

### ✅ Data Ingestion Module

- Read the raw dataset (`stud.csv`) from the specified location
- Save a backup copy of the raw data to `artifacts/raw.csv`
- Perform a 70/30 train-test split using `train_test_split`
- Store the resulting datasets in:
  - `artifacts/train.csv`
  - `artifacts/test.csv`
- Modular and reusable class-based design using `@dataclass` for configuration

> 📂 `src/components/data_ingestion.py`

---

### ✅ Data Transformation Module

- Handles missing values (if any)
- Encodes categorical features
- Scales numerical columns using `StandardScaler`
- Saves the transformation pipeline (`preprocessor.pkl`) for reuse in prediction

> 📂 `src/components/data_transformation.py`

---

### ✅ Model Training & Evaluation

- Trains multiple models (e.g., Random Forest, Linear Regression)
- Evaluates on test data using R², MAE, MSE
- Selects best-performing model and saves it as `model.pkl`
- Modular training logic with performance logging

> 📂 `src/components/model_trainer.py`

---

### ✅ Predict Pipeline

- Uses saved model and transformer (`model.pkl`, `preprocessor.pkl`)
- Loads input features and outputs prediction
- Designed to work with both backend services and web inputs

> 📂 `src/pipelines/predict_pipeline.py`

---

### ✅ Frontend (Basic HTML/CSS)

- A minimal HTML/CSS form to collect user inputs (features)
- Sends input data to the backend for prediction
- Displays the predicted performance score/output to the user

> 📂 `templates/index.html`  
> 📂 `static/style.css`

--- 

### ✅ Dockerization

- Fully containerized using a custom `Dockerfile`
- Deployed on DockerHub: [https://hub.docker.com/repository/docker/bengj/stud-perf-proj/general]
- Runs the app in a consistent environment using:

```bash
docker build -t student-perf-ml .
docker run -p 5000:5000 student-perf-ml
```


> 📂 `Dockerfile`
--- 

## 2. Technologies Used

- Python 3.8+

- Pandas, NumPy

- Scikit-learn

- Logging & Exception Handling

- Dataclasses

- HTML & CSS (Frontend)

- Docker

---

## 3. Next Steps

1. Add CI/CD using GitHub Actions

2. Cloud Deployment (Render, Railway, or AWS)

---

## 4. Author
Developed by Ben Gregory John

BTech CSE | 2nd Year

GitHub: [https://github.com/BenGJ10/]
