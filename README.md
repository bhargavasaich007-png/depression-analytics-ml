# 🧠 Depression Analytics and Machine Learning

## 🔍 Overview
A machine learning project to analyse workplace mental health data and predict
depression risk. The project covers the full data analytics pipeline including
descriptive analysis, classification, regression, association rule mining, and
clustering — all implemented in Python.

## 📊 Dataset
| Detail | Info |
|--------|------|
| File | `Depression_Professional_Dataset.csv` |
| Rows | 2,054 |
| Columns | 11 |
| Missing Values | None |

**Target Variable:** `Depression` (0 = No, 1 = Yes)

**Features:** Age, Gender, Work Pressure, Job Satisfaction, Sleep Duration,
Dietary Habits, Suicidal Thoughts, Work Hours, Financial Stress,
Family History of Mental Illness

> ⚠️ Class Imbalance: Only 10% positive cases — Random Oversampling applied.

## 🛠️ Tools & Technologies
- **Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn,
  Imbalanced-learn, MLxtend, SciPy
- **Environment:** Jupyter Notebook

## ⚙️ Approach
1. Descriptive Analysis — distributions, visualisations, class imbalance check
2. Classification — Logistic Regression, SVM, Decision Tree with oversampling
3. Regression — Linear, Ridge, Random Forest with label encoding and normalisation
4. Association Rule Mining — Apriori algorithm with one-hot encoding
5. Clustering — K-Means and Hierarchical Clustering with PCA visualisation

## 📈 Results
| Task | Model | Result |
|------|-------|--------|
| Classification | Logistic Regression | Accuracy = 95% |
| Classification | SVM (RBF, tuned) | Accuracy = 100% |
| Classification | Decision Tree | Accuracy = 100% |
| Regression | Linear Regression | R² = 0.25 |
| Regression | Ridge Regression | R² = 0.25 |
| Regression | Random Forest | R² = 0.16 |
| Association Rules | Apriori | 2017 rules, top confidence = 100% |
| Clustering | K-Means (k=2) | Silhouette Score = 0.102 |

## 📁 Files
| File | Description |
|------|-------------|
| `notebooks/depression_analytics_all_tasks.ipynb` | Main Jupyter notebook |
| `data/Depression_Professional_Dataset.csv` | Dataset |
| `report/GROUP_129_Report.pdf` | Full project report |
| `requirements.txt` | Python libraries to install |

## 🔮 Future Work
- SMOTE instead of Random Oversampling for better class balancing
- Deep learning model for improved classification
- Feature importance analysis using SHAP values
- Deploy as a web app using Flask or Streamlit
