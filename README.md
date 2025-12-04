# CAI-5724-Edgar-Diabetes Prediction Using Machine Learning & Deep Learning

A comprehensive machine learning project comparing **5 different algorithms** (traditional ML + deep learning) for predicting diabetes risk using the Pima Indians Diabetes dataset.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Why This Dataset?](#why-this-dataset)
- [What This Code Does](#what-this-code-does)
- [Why This Project is Useful](#why-this-project-is-useful)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

Early detection of diabetes is critical for preventing serious complications like heart disease, kidney failure, and blindness. This project demonstrates how **artificial intelligence** can assist in predicting diabetes risk using readily available clinical measurements.

**Key Question:** Can machine learning accurately predict diabetes from 8 simple medical features?

**Models Compared:**
1. Logistic Regression (baseline)
2. Random Forest Classifier
3. XGBoost (Gradient Boosting)
4. Support Vector Machine (SVM)
5. **Neural Network** (5-layer deep learning model)

---

## 🏥 Why This Dataset?

### Dataset: Pima Indians Diabetes Database
- **Source:** National Institute of Diabetes and Digestive and Kidney Diseases
- **Patients:** 768 Pima Indian women (age 21+)
- **Features:** 8 clinical measurements
- **Target:** Binary classification (Diabetes: Yes/No)

### Why Pima Indians?
The Pima Indian population has one of the **highest diabetes rates in the world** (~50% prevalence), making them an important population for diabetes research. This dataset has been extensively studied and validated in medical literature.

### Why This Dataset is Ideal for Learning:
✅ **Real medical data** - Not synthetic  
✅ **Clean and well-documented** - Industry standard benchmark  
✅ **Right size** - 768 samples (not too big, not too small)  
✅ **Balanced complexity** - 8 features (manageable but non-trivial)  
✅ **Clear outcome** - Binary classification (diabetes yes/no)  
✅ **Clinical relevance** - Features align with actual medical practice  

### Features (Input Variables):
| Feature | Description | Clinical Significance |
|---------|-------------|----------------------|
| **Pregnancies** | Number of pregnancies | Gestational diabetes risk factor |
| **Glucose** | Plasma glucose concentration | Primary diabetes indicator |
| **BloodPressure** | Diastolic blood pressure (mm Hg) | Cardiovascular health |
| **SkinThickness** | Triceps skin fold thickness (mm) | Body fat estimation |
| **Insulin** | 2-Hour serum insulin (mu U/ml) | Insulin resistance indicator |
| **BMI** | Body mass index (kg/m²) | Obesity measure |
| **DiabetesPedigreeFunction** | Genetic predisposition score | Family history |
| **Age** | Age in years | Age-related risk |

---

## 🔬 What This Code Does

### 1. Data Exploration & Preprocessing
- Statistical analysis of patient demographics
- Visualization of feature distributions
- Correlation analysis between features
- Handling missing values (zeros coded as missing)
- Outlier detection using IQR method
- Feature scaling (standardization)

### 2. Model Training & Comparison
- Trains **5 different algorithms** on the same data
- Uses **5-fold cross-validation** for robust evaluation
- Performs **hyperparameter tuning** on Random Forest
- Builds custom **neural network** with:
  - 3 hidden layers (64 → 32 → 16 neurons)
  - Dropout regularization (prevents overfitting)
  - Batch normalization (faster training)
  - Early stopping (automatic convergence)

### 3. Comprehensive Evaluation
Evaluates models using **multiple metrics**:
- **Accuracy** - Overall correctness
- **ROC AUC** - Discrimination ability (0.8-0.9 = excellent)
- **Sensitivity (Recall)** - % of diabetes cases caught
- **Specificity** - % of healthy patients correctly identified
- **Precision** - % of positive predictions that are correct
- **F1-Score** - Harmonic mean of precision/recall

### 4. Advanced Visualizations
- **Confusion matrices** - Show prediction breakdown (TP, TN, FP, FN)
- **ROC curves** - Compare discrimination across models
- **Precision-Recall curves** - Performance on imbalanced data
- **Feature importance** - Identify most predictive variables
- **Training history** - Neural network learning curves
- **Model comparison charts** - Side-by-side performance

### 5. Model Explainability (SHAP)
- SHAP values for interpretable AI
- Shows how each feature contributes to individual predictions
- Validates that model decisions align with medical knowledge

---

## 💡 Why This Project is Useful

### For Students & Learners:
✅ **Complete ML workflow** - From raw data to production-ready models  
✅ **Best practices** - Proper train/test split, cross-validation, stratification  
✅ **Multiple algorithms** - Learn when to use each approach  
✅ **Deep learning** - Hands-on neural network implementation  
✅ **Well-commented** - Every line explained with clinical context  
✅ **Visualization** - Professional plots for presentations  

### For Healthcare Professionals:
✅ **Clinical validation** - Model decisions match medical knowledge  
✅ **Interpretable results** - Not a black box  
✅ **Multiple metrics** - Understand trade-offs (sensitivity vs specificity)  
✅ **Real-world applicable** - Uses standard clinical measurements  

### For Data Scientists:
✅ **Benchmark comparison** - Test your own models against these baselines  
✅ **Preprocessing techniques** - Handle missing data, outliers, scaling  
✅ **Model evaluation** - Go beyond accuracy with ROC, PR curves  
✅ **Imbalanced data handling** - 65/35 class split (realistic scenario)  
✅ **Hyperparameter tuning** - Grid search demonstration  

### For Researchers:
✅ **Reproducible** - Fixed random seeds, clear methodology  
✅ **Extensible** - Easy to add new features or models  
✅ **Well-documented** - Clear explanations of design choices  
✅ **Standard dataset** - Compare with published literature  
