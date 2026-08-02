# Credit Card Fraud Detection — End-to-End ML Pipeline on Databricks

An end-to-end machine learning project built on **Databricks**, covering the full lifecycle from raw data ingestion to a live, queryable **Model Serving Endpoint**. The project detects fraudulent credit card transactions using PySpark MLlib.

## 🎯 Project Overview

Credit card fraud is a classic imbalanced classification problem — fraudulent transactions make up a tiny fraction of all transactions, but missing one is costly. This project builds, evaluates, and deploys a fraud detection model using real anonymized transaction data.

## 📊 Dataset

- **Source:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle, ULB Machine Learning Group)
- **File:** `Dataset/creditcard.csv.gz` — Extract with `gunzip creditcard.csv.gz` before use
- **Size:** 284,807 transactions, 31 columns
- **Features:** `Time`, `V1`–`V28` (PCA-transformed, anonymized), `Amount`
- **Target:** `Class` (0 = genuine, 1 = fraud)
- **Class imbalance:** 492 frauds (0.17%) vs 284,315 genuine transactions

## 🏗️ Architecture

```
Kaggle Dataset
      │
      ▼
Databricks Unity Catalog Volume (raw CSV storage)
      │
      ▼
PySpark DataFrame (load + explore + clean)
      │
      ▼
Feature Engineering (StandardScaler on Time & Amount, VectorAssembler)
      │
      ▼
Train/Test Split (80/20)
      │
      ▼
Model Training (Logistic Regression, Random Forest — PySpark MLlib)
      │
      ▼
MLflow Tracking (params, metrics, model artifact, signature)
      │
      ▼
Unity Catalog Model Registry
      │
      ▼
Databricks Model Serving Endpoint (real-time REST API)
```

## 🔧 Tech Stack

- **Platform:** Databricks (Free Edition, Serverless compute)
- **Processing:** PySpark, Spark MLlib
- **Experiment Tracking:** MLflow
- **Model Registry:** Unity Catalog
- **Deployment:** Databricks Model Serving (REST endpoint)
- **Storage:** Unity Catalog Volumes

## 📈 Data Preprocessing

- Loaded raw CSV from a Unity Catalog Volume into a Spark DataFrame
- Verified data quality: no missing values across all 31 columns
- `V1`–`V28` were already PCA-transformed (roughly -5 to +5 range) — no scaling needed
- `Time` and `Amount` were on very different scales, so both were standardized using `StandardScaler` (mean=0, std=1)
- Assembled all 30 features (`V1`–`V28`, scaled `Time`, scaled `Amount`) into a single feature vector using `VectorAssembler`
- Split into 80% training (228,029 rows) / 20% test (56,778 rows)

## 🤖 Models Trained & Compared

| Model | AUC | Frauds Caught | Frauds Missed | False Alarms |
|---|---|---|---|---|
| Logistic Regression (unweighted) | 0.9906 | 57 / 87 (65.5%) | 30 | 6 |
| Logistic Regression (class-weighted) | 0.9880 | 85 / 87 (97.7%) | 2 | 1,400 |
| **Random Forest (100 trees, unweighted)** | **0.9930** | **69 / 87 (79.3%)** | **18** | **8** |

**Why accuracy alone is misleading:** with 99.8% of transactions being genuine, a model predicting "genuine" for everything scores ~99.8% accuracy while catching zero fraud. This project instead evaluates using **AUC, precision, recall, and the confusion matrix**.

**Final model: Random Forest** (`numTrees=100`, `maxDepth=10`) — selected for the best balance of high fraud recall and low false-positive rate, without the precision collapse seen in the weighted Logistic Regression approach.

## 🚀 Deployment

1. Model logged to **MLflow** with parameters, metrics, and an inferred input/output signature
2. Registered in the **Unity Catalog Model Registry** as `workspace.default.credit_card_fraud_rf`
3. Deployed to a **Databricks Model Serving Endpoint** (CPU, serverless, scale-to-zero enabled)
4. Tested via REST API — returns real-time fraud predictions (`0` = genuine, `1` = fraud) given a transaction's feature vector

**Example request:**
```json
{
  "dataframe_split": {
    "columns": ["features"],
    "data": [[[-1.359, -0.072, 2.536, ...]]]
  }
}
```

**Example response:**
```json
{
  "predictions": [0.0]
}
```

## 📁 Project Structure

```
├── notebooks/
│   └── credit_card_fraud.ipynb   # Full pipeline: load → EDA → preprocess → train → evaluate → deploy
├── README.md
```

## 🔮 Future Improvements

- Try Gradient Boosted Trees / XGBoost for comparison
- Tune the classification threshold instead of relying on class weights, to better balance precision/recall for a specific business cost tradeoff
- Add SHAP-based feature importance analysis
- Enable inference tables on the serving endpoint for production monitoring
- Add automated retraining via a Databricks Job

## 📄 License

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), released under the Open Database License (ODbL) by the Machine Learning Group at ULB.

---

## 📬 Contact

Haris Majeed — [harismajeed299@gmail.com](mailto:harismajeed299@gmail.com)

> For More Projects: [**Follow me on GitHub**](https://github.com/Harris-Majeed)
