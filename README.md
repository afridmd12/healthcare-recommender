
# 🏥 Personalized Healthcare Recommendation System

### **Download Links for Required Technologies**
- 🐍 **Python** → [Download Python 3.9+](https://www.python.org/downloads/)
- 🌐 **Flask** → [Flask Official Documentation](https://flask.palletsprojects.com/en/latest/)
- ⚡ **XGBoost** → [XGBoost Installation Guide](https://xgboost.readthedocs.io/en/stable/install.html)


## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Objectives](#objectives)
- [Approach & Methods](#approach--methods)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Setup & Usage](#setup--usage)
- [Requirements (inline)](#requirements-inline)
- [Visualizations](#visualizations)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Contact](#contact)

---

## Overview
The **Personalized Healthcare Recommendation System** is an end-to-end machine learning application that predicts patient health conditions and provides actionable recommendations. It uses **XGBoost for classification**, handles **class imbalance using SMOTE**, and offers predictions via a **Flask web interface**.

---

## Dataset
- **File:** `blood.csv`
- **Rows × Columns:** 748 × 5
- **Features:**
  - `Recency` (days since last visit)
  - `Frequency` (number of visits)
  - `Monetary` (amount spent)
  - `Time` (relationship duration)
- **Target:**
  - `target` (0 = Healthy, 1 = Requires Attention)

---

## Objectives
1. Preprocess and scale data.
2. Handle imbalance with SMOTE.
3. Train XGBoost classifier for predictions.
4. Build Flask web app for real-time recommendations.

---

## Approach & Methods
- **Preprocessing:** Scaled numeric features with StandardScaler.
- **Imbalance Handling:** SMOTE applied to balance classes.
- **Modeling:** XGBoost Classifier (`eval_metric='mlogloss'`), train/test split 80/20.
- **Evaluation:** Accuracy, Precision, Recall, F1-score.

---

## Results
- **Before SMOTE:** Class 0 = 570, Class 1 = 178.
- **Performance:** *(Example values)*
  - Accuracy: ~0.76
  - Precision: ~0.7512
  - Recall: ~0.76
  - F1-score: ~0.755

---

## Repository Structure
```
health_app/
├─ app.py               # Flask app
├─ model.pkl            # Trained model
├─ scaler.pkl           # Scaler
├─ templates/
│   └── index.html      # Web UI
├─ healthcare_blood.ipynb # Notebook
├─ blood.csv            # Dataset
└─ README.md            # Documentation
```

---

## Setup & Usage
### 1. Clone Repository
```bash
git clone https://github.com/afridmd12/healthcare-recommender.git
cd healthcare-recommendation
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run Application
```bash
python app.py
```
Visit: **http://127.0.0.1:5000/**

---

## Requirements (inline)
```
flask
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
```

---

## Visualizations
- Feature Importance (XGBoost)
- Confusion Matrix
- ROC Curve

---

## Limitations
- Limited features; add more for better accuracy.

---

## Future Work
- Add SHAP explainability.
- Deploy on Render / Hugging Face.
- Include diet and fitness advice.

---

## Acknowledgements
- Libraries: Flask, XGBoost, pandas, scikit-learn, imbalanced-learn.

---
