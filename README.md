# Credit Card Fraud Detection

> A reproducible repository for building, evaluating, and deploying machine learning models to detect credit card fraud.

---

## 🔍 Project Overview

This repository contains code, notebooks, and model artifacts for training and evaluating fraud-detection systems on transactional data. The goal is to create an explainable, well-evaluated pipeline that addresses severe class imbalance and prioritizes precision, recall, and real-world deployability.

Typical contents:

* Data preparation and feature engineering
* Model training (classical ML + tree-based + deep learning)
* Evaluation metrics and visualization (ROC, PR curve, confusion matrix)
* Baselines and experiments (undersampling, oversampling, class weights)
* Example notebook for model explainability (SHAP/feature importance)

---

## ⚙️ Features

* Fully reproducible training pipeline (scripts + notebooks).
* Built-in support for class imbalance techniques: class weighting, SMOTE, undersampling.
* Ready-to-run examples: Logistic Regression, Random Forest, XGBoost/LightGBM, and a small neural network.
* Model evaluation with business-oriented metrics (precision-at-k, cost-based metrics).
* Notebook showing explainability using SHAP values.

---

## 📁 Repository Structure

```
├── data/                  # (ignored) raw and processed datasets
├── notebooks/             # exploratory notebooks and tutorials
│   ├── 01-data-exploration.ipynb
│   ├── 02-preprocessing.ipynb
│   └── 03-modeling-and-evaluation.ipynb
├── src/                   # python package: data, features, models, utils
│   ├── data/              # data loading and preprocessing
│   ├── features/          # feature engineering pipelines
│   ├── models/            # training and inference code
│   └── evaluation/        # metrics and plotting utilities
├── scripts/               # executable scripts for training & evaluating
│   ├── train.py
│   └── evaluate.py
├── requirements.txt
├── environment.yml        # optional conda environment
├── Dockerfile             # optional containerization for deployment
├── README.md              # this file
└── LICENSE
```

---

## 🧰 Quickstart — Installation

```bash
# clone
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# create venv
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate     # Windows

# install
pip install -r requirements.txt
```

If you prefer conda:

```bash
conda env create -f environment.yml
conda activate fraud-detect
```

---

## 🗃️ Data

> **Note:** This repo does not include proprietary bank data. Use the dataset you are authorized to access.

A commonly used public dataset is the *Credit Card Fraud Detection* dataset from Kaggle (European cardholders) — it contains anonymized numerical features and a `Class` label where `1` indicates fraud. Download it and place the `creditcard.csv` file in `data/raw/`.

Data pipeline steps in `src/data`:

* load raw CSV
* basic cleaning (missing values, type checks)
* scaling (StandardScaler or RobustScaler)
* optional PCA for dimensionality reduction
* produce `data/processed/train.csv` and `data/processed/test.csv`

---

## 🧩 Preprocessing & Feature Engineering

Common steps implemented in the repo:

* Outlier checks and capping
* Time-based features (if available)
* Aggregations by `user_id` or `card_id` (if present)
* Encoding categorical variables (target/one-hot)
* Resampling strategies: RandomUnderSampler, SMOTE, SMOTEENN
* Pipeline compatibility with scikit-learn `Pipeline` API

---

## 🔬 Modeling

Examples included:

* Baseline: Logistic Regression (with class weights)
* Tree-based: Random Forest, XGBoost, LightGBM
* Neural network: small MLP with early stopping

Training scripts accept common CLI args (config file support recommended):

```bash
python scripts/train.py --config configs/experiment.yaml
```

---

## 📈 Evaluation & Metrics

Due to extreme class imbalance, accuracy is often misleading. Use the following:

* Precision, Recall, F1-score
* ROC AUC
* PR AUC (Precision-Recall AUC) — better for rare events
* Confusion matrix with thresholding
* Precision@k (precision among top k suspicious transactions)
* Cost-based metric (custom: cost of false negative vs false positive)

The `src/evaluation` module contains plotting helpers for ROC, PR curves, and threshold analysis.

---

## ♻️ Example: train & evaluate

Train a baseline Random Forest with class weighting:

```bash
python scripts/train.py --model random_forest --data data/processed --output artifacts/rf
python scripts/evaluate.py --model-path artifacts/rf/model.pkl --test data/processed/test.csv
```

You can run the example notebooks interactively to see model outputs and visualizations.

---

## 🔍 Explainability

We include a notebook using SHAP to explain predictions from tree models. This helps surface which features most influence the fraud decision — useful for analysts and auditors.

---

## 🛡️ Ethical considerations

* Fraud datasets often contain sensitive information. Only use data you are authorized to access and follow your organization’s privacy policies.
* Models can produce biased or brittle decisions. Use human-in-the-loop review for flagged transactions.
* Log model decisions and inputs for auditability, while respecting privacy.

---

## ✅ Contribution

Contributions are welcome. If you'd like to add a model, experiment, or fix, please open an issue or PR. A suggested workflow:

1. Fork the repo
2. Create a feature branch
3. Add tests where applicable
4. Open a PR with a clear description of changes

---

## 📦 Requirements (example)

```text
python>=3.9
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
xgboost
lightgbm
shap
joblib
jupyterlab
```

(See `requirements.txt` for exact pinned versions.)

---

## 📜 License

This project is released under the MIT License. See `LICENSE` for details.

---

## 🧾 Notes

If you want a shorter README (repo landing page) or a longer `CONTRIBUTING.md` and `PIPELINE.md`, tell me which parts you want trimmed or expanded. I can also:

* generate `requirements.txt` for you
* add a `Makefile` or GitHub Actions CI workflow
* scaffold a Dockerfile and simple REST inference service

---

*Made with ❤️ — tell me if you want the README tuned for a banking production environment, Kaggle competition, or academic project.*
