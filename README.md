```markdown
# 🌾 Predicting Irrigation Need | Kaggle Playground Series (S6E4)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)
![Ensemble Models](https://img.shields.io/badge/Models-Ensemble_v3-green.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-Playground%20S6E4-blueviolet.svg)

## 📌 Overview
This repository contains an end-to-end Machine Learning pipeline developed for the **Kaggle Playground Series (Season 6, Episode 4)** competition. The goal of this challenge is to accurately predict crop **irrigation needs** (`Low`, `Medium`, `High`) using multi-modal tabular datasets containing soil parameters, climate conditions, and crop characteristics.

---

## 📁 Repository Structure

```text
predicting-irrigation-need/
│
├── data/                            # Raw and processed datasets
├── image/                           # Visualizations, charts, and EDA plots
├── notebooks/                       # Exploratory Data Analysis & experiments
├── templates/                       # Web UI/Deployment templates (Flask/FastAPI)
│
├── categories_map.pkl               # Saved categorical features mapping
├── model.pkl                        # Trained final ensemble model artifact
├── other.ipynb                      # Additional pipeline and modeling scripts
│
├── submission_98_final.csv          # Final prediction submission
├── submission_elite_ensemble_v3.csv # Elite Ensemble predictions
└── submission_final_threshold_...   # Threshold-tuned predictions

```

---

## 📊 Dataset & Exploratory Findings

* **Data Volume:** 630,000 training samples and 270,000 testing samples.
* **Missing Values & Duplicates:** 0 missing values across all 21 attributes; 0 duplicate rows.
* **Class Imbalance:**
* `Low`: **58.72%**
* `Medium`: **37.95%**
* `High`: **3.33%** *(Addressed using stratified sampling and custom decision thresholds)*



---

## 🛠️ Project Pipeline

1. **Data Preprocessing & Feature Engineering:**
* Feature mapping via `categories_map.pkl` for consistent categorical encoding.
* Interaction features combining soil moisture, rainfall, and temperature variables.


2. **Modeling & Ensembling:**
* Model exploration including LightGBM, XGBoost, and CatBoost.
* Elite Ensembling strategy to boost multi-class log-loss performance.
* Probability threshold tuning (`submission_final_threshold_...csv`) to optimize minority class (`High`) classification.


3. **Deployment Assets:**
* Serialized model artifact (`model.pkl`) for inference.
* HTML/UI templates in `templates/` for application integration.



---

## 🚀 Getting Started

1. **Clone the Repository:**
```bash
git clone [https://github.com/your-username/predicting-irrigation-need.git](https://github.com/your-username/predicting-irrigation-need.git)
cd predicting-irrigation-need

```


2. **Load the Model Artifact in Python:**
```python
import pickle

# Load categorical encoding mapping
with open('categories_map.pkl', 'rb') as f:
    cat_map = pickle.load(f)

# Load trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

```

