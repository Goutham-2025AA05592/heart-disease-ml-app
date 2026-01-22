# Heart Disease Prediction using Machine Learning

---

## 1. Problem Statement
Heart disease is one of the leading causes of mortality worldwide. Early detection using patient clinical attributes can assist healthcare professionals in making informed decisions and initiating preventive care.

The goal of this project is to **predict the presence of heart disease** using supervised machine learning techniques based on patient health parameters.  
This system is intended as a **decision-support tool** and **not a replacement for professional medical diagnosis**.

---

## 2. Dataset Description
The dataset used in this project is the **Heart Disease UCI Dataset**, sourced from [Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data).

- **Number of instances:** 920  
- **Number of features:** 15
- **Number of effective features used:** 13
- **Target variable:** `num`  
  - `0` → No heart disease  
  - `1` → Presence of heart disease (all values > 0 converted to 1)

The dataset aggregates patient records from multiple medical centers and contains a mix of numerical and categorical clinical attributes.

### Key Features

| Feature | Description |
|---------|-------------|
| age | Age of the patient |
| sex | Gender (Male/Female) |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina (Yes/No) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels (0–3) |
| thal | Thalassemia (normal, fixed defect, reversible defect) |

---

## 3. Data Preprocessing Pipeline
To ensure consistency between training and inference, the following preprocessing steps were applied:

1. **Column Removal:** `id` and `dataset` columns were dropped to prevent data leakage and source bias.  
2. **Missing Value Handling:**  
   - Numerical features → Median imputation  
   - Categorical features → Mode imputation  
3. **One-Hot Encoding** for categorical variables.  
4. **Feature Scaling** using **StandardScaler**.  
5. **Binary Classification:** `num > 0` converted to class `1`.  
6. **Feature Alignment:** During inference, uploaded datasets are aligned to the trained model feature set using saved `feature_names.pkl`.

---

## 4. Machine Learning Models Implemented
The following six classification models were implemented using the same dataset and preprocessing pipeline:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)

---

## 5. Model Evaluation Metrics
Each model was evaluated on a held-out test dataset using the following metrics:

- **Accuracy** – Overall correctness  
- **AUC Score** – Ability to discriminate between classes  
- **Precision** – True positives / predicted positives  
- **Recall** – True positives / actual positives  
- **F1 Score** – Harmonic mean of precision and recall  
- **Matthews Correlation Coefficient (MCC)** – Robust evaluation in case of class imbalance

### Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------|--------|-----|-----------|--------|--------|-----|
| Logistic Regression | 0.8424 | 0.9032 | 0.8411 | 0.8824 | 0.8612 | 0.6801 |
| Decision Tree | 0.7935 | 0.7886 | 0.8019 | 0.8333 | 0.8173 | 0.5806 |
| K-Nearest Neighbors (KNN) | 0.8533 | 0.9077 | 0.8505 | 0.8922 | 0.8708 | 0.7023 |
| Naive Bayes | 0.8424 | 0.8869 | 0.8544 | 0.8627 | 0.8585 | 0.6807 |
| Random Forest | 0.8696 | 0.9211 | 0.8482 | 0.9314 | 0.8879 | 0.7374 |
| XGBoost | 0.8587 | 0.8827 | 0.8455 | 0.9118 | 0.8774 | 0.7141 |

---
## 6. Observations on Model Performance

| Model | Observation |
|------|-------------|
| **Logistic Regression** | Demonstrates a strong and interpretable baseline with balanced precision (**0.8411**) and recall (**0.8824**). The high AUC (**0.9032**) indicates good class separability, making it suitable for explainable healthcare applications. |
| **Decision Tree** | Achieves reasonable recall (**0.8333**) but records the lowest accuracy (**0.7935**) and MCC (**0.5806**), indicating overfitting and weaker generalization when used independently. |
| **K-Nearest Neighbors (KNN)** | Performs effectively after feature scaling, achieving balanced precision (**0.8505**) and recall (**0.8922**) with a strong MCC (**0.7023**), highlighting the importance of normalization for distance-based classifiers. |
| **Naive Bayes** | Maintains stable performance despite strong independence assumptions. With an F1 score of (**0.8585**) and MCC (**0.6807**), it serves as a fast and reliable probabilistic baseline. |
| **Random Forest** | Delivers the best overall performance, achieving the highest accuracy (**0.8696**), AUC (**0.9211**), recall (**0.9314**), F1 score (**0.8879**), and MCC (**0.7374**). Ensemble learning effectively captures complex feature interactions while reducing overfitting. |
| **XGBoost** | Performs competitively with strong recall (**0.9118**) and F1 score (**0.8774**). Although its MCC (**0.7141**) is slightly lower than Random Forest, it remains highly effective for identifying positive cases in healthcare scenarios. |

### Key Takeaways

1. **Ensemble models outperform individual classifiers**, with **Random Forest** emerging as the most reliable model across evaluation metrics.
2. **Recall is prioritized in healthcare applications**, and ensemble methods achieve high recall without significantly sacrificing precision.
3. **Feature preprocessing is essential**, particularly for Logistic Regression and KNN, which are sensitive to feature scaling.
4. **Simple models remain valuable**, offering interpretability and stable baselines.
5. **Single decision trees exhibit generalization limitations**, reinforcing the advantage of ensemble techniques.
---

## 7. Streamlit Web Application
An interactive **Streamlit web application** was developed to demonstrate real-world usage of the trained models.

### Features
- Upload CSV test dataset  
- Download sample test dataset  
- Sidebar-based model selection  
- Automatic preprocessing aligned with training pipeline  
- KPI-style metric cards for intuitive interpretation  
- Confusion matrix visualization  
- ROC curve visualization  
- Tab-based layout:
  - **Model Performance**  
  - **Benchmarking** (dynamic comparison using session state)  
  - **Data Exploration** (numerical stats, correlation heatmap, categorical counts, target distribution)  

The application ensures **feature consistency** by aligning uploaded data with the trained model feature set.

---

## 8. Project Structure

```text
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- train_and_save_models.py
│
│-- data/
│   ├── heart_disease_test_sample_500.csv
│   └── heart_disease_uci.csv
│
│-- model/
│   ├── logistic_regression_model.pkl
│   ├── logistic_regression_model.py
│   ├── decision_tree_model.pkl
│   ├── decision_tree_model.py
│   ├── knn_model.pkl
│   ├── knn_model.py
│   ├── naive_bayes_model.pkl
│   ├── naive_bayes_model.py
│   ├── random_forest_model.pkl
│   ├── random_forest_model.py
│   ├── xgboost_model.pkl
│   ├── xgboost_model.py
│   ├── feature_names.pkl
│   ├── scaler.pkl
│   └── model_metrics.csv

```
This structure separates **data, model training, and deployment components** to ensure clarity, reproducibility, and maintainability

---

## 9. Deployment
The Streamlit application is deployed using **Streamlit Community Cloud** and is accessible via a public link.  
All dependencies are managed using `requirements.txt` to ensure smooth deployment.

---

## 10. Links
🔗 **Live Streamlit App:** *https://heart-disease-ml-app-4pbhzzcrmnvs7hgarsmfvx.streamlit.app/*  
🔗 **GitHub Repository:** *https://github.com/Goutham-2025AA05592/heart-disease-ml-app*  

---

## 11. Assignment Details
- **Student ID:** 2025AA05592
- **Program:** M.Tech (AIML)  
- **Course:** Machine Learning  
- **Institution:** BITS Pilani  

---
