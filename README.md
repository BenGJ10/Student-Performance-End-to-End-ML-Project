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

> Refer to: `src/logger.py`

---

### ✅ Custom Exception Handling

- Centralized error management using a `CustomException` class
- Catches and formats exceptions with traceback and error context
- Used across the ingestion module for better fault tolerance

> Refer to: `src/exception.py`

---

### ✅ Data Ingestion Module

- Read the raw dataset (`stud.csv`) from the specified location
- Save a backup copy of the raw data to `artifacts/raw.csv`
- Perform a 70/30 train-test split using `train_test_split`
- Store the resulting datasets in:
  - `artifacts/train.csv`
  - `artifacts/test.csv`
- Modular and reusable class-based design using `@dataclass` for configuration

> Refer to: `src/components/data_ingestion.py`

---

## 2. Technologies Used
- Python 3.8+

- Pandas

- Scikit-learn

- Logging

- Dataclasses

- Custom Exception Handling

---

## 3. Next Steps

The following modules will be implemented as part of the complete pipeline:

1. Data Transformation (handling missing values, encoding, scaling)

2. Model Training and Evaluation

3. Lot more to uncover

---

## 4. Author
Developed by [Ben Gregory John]

BTech CSE | 2nd Year

GitHub: [https://github.com/BenGJ10/]
