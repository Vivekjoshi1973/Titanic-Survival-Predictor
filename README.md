# Titanic Survival Predictor

A Streamlit app that predicts whether a Titanic passenger survived. Five models were compared on the classic Titanic dataset; **SVM (RBF kernel)** was chosen as the final model and deployed.

**Live app:** https://titanic-survival-predictor-vj.streamlit.app/

## Features

- Ticket class, sex, age, siblings/spouses, parents/children, fare, port of embarkation, and travelling-alone inputs
- One-click survival prediction
- Model cached with `st.cache_resource` for fast load times

## Project structure

```
Titanic Survival Predictor/
├── Models/
│   └── titanic_model.pkl      # trained SVM (RBF) pipeline: StandardScaler + SVC
├── Notebook/
│   └── titanic.ipynb          # EDA and model training
├── Streamlit_App/
│   └── app.py                 # Streamlit app
├── requirements.txt
└── README.md
```

## Models evaluated

All five models were trained and tested on the same 80/20 split (`random_state=42`, 178 test samples). Dataset: `sns.load_dataset("titanic")` (889 rows after dropping missing `embarked`), features: `pclass, sex, age, sibsp, parch, fare, embarked, alone`.

| Model | Accuracy | Precision (class 0 / 1) | Recall (class 0 / 1) | F1 (class 0 / 1) |
|---|---|---|---|---|
| **SVM (RBF kernel)** | **82.58%** | 0.84 / 0.80 | 0.88 / 0.74 | 0.86 / 0.77 |
| Logistic Regression | 80.34% | 0.85 / 0.74 | 0.83 / 0.77 | 0.84 / 0.75 |
| KNN (k=5) | 78.09% | 0.82 / 0.72 | 0.83 / 0.71 | 0.82 / 0.72 |
| Naive Bayes (Gaussian) | 77.53% | 0.85 / 0.68 | 0.77 / 0.78 | 0.81 / 0.73 |
| Decision Tree | 75.84% | 0.81 / 0.69 | 0.80 / 0.70 | 0.80 / 0.69 |

### Classification reports (test set, 178 samples: 109 not survived, 69 survived)

**SVM (RBF kernel) — 82.58% (highest)**

```
              precision    recall  f1-score   support
           0       0.84      0.88      0.86       109
           1       0.80      0.74      0.77        69
    accuracy                           0.83       178
   macro avg       0.82      0.81      0.81       178
weighted avg       0.82      0.83      0.82       178
```

Confusion matrix: `[[96, 13], [18, 51]]`

**Logistic Regression — 80.34%**

```
              precision    recall  f1-score   support
           0       0.85      0.83      0.84       109
           1       0.74      0.77      0.75        69
    accuracy                           0.80       178
   macro avg       0.79      0.80      0.79       178
weighted avg       0.81      0.80      0.80       178
```

Confusion matrix: `[[90, 19], [16, 53]]`

**KNN (k=5) — 78.09%**

```
              precision    recall  f1-score   support
           0       0.82      0.83      0.82       109
           1       0.72      0.71      0.72        69
    accuracy                           0.78       178
   macro avg       0.77      0.77      0.77       178
weighted avg       0.78      0.78      0.78       178
```

Confusion matrix: `[[90, 19], [20, 49]]`

**Naive Bayes (Gaussian) — 77.53%**

```
              precision    recall  f1-score   support
           0       0.85      0.77      0.81       109
           1       0.68      0.78      0.73        69
    accuracy                           0.78       178
   macro avg       0.77      0.78      0.77       178
weighted avg       0.78      0.78      0.77       178
```

Confusion matrix: `[[84, 25], [15, 54]]`

**Decision Tree — 75.84%**

```
              precision    recall  f1-score   support
           0       0.81      0.80      0.80       109
           1       0.69      0.70      0.69        69
    accuracy                           0.76       178
   macro avg       0.75      0.75      0.75       178
weighted avg       0.76      0.76      0.76       178
```

Confusion matrix: `[[87, 22], [21, 48]]`

### Why SVM was chosen

- **Highest accuracy**: 82.58% vs 80.34% (Logistic Regression), 78.09% (KNN), 77.53% (Naive Bayes), 75.84% (Decision Tree).
- **Best F1 for both classes** (0.86 not-survived, 0.77 survived) — the most balanced precision/recall trade-off of all five models.
- **Strongest detection of non-survivors** (recall 0.88) while keeping survivor precision at 0.80, so it makes fewer dangerous false "survived" predictions.
- Trained inside a `Pipeline(StandardScaler, SVC(kernel="rbf"))`, so scaling is applied automatically at inference time — no manual preprocessing needed by the app.
- Decision Trees and KNN are more prone to overfitting/variance on this small dataset; SVM with the RBF kernel generalizes better here.

## Setup and run locally

```bash
pip install -r requirements.txt
streamlit run Streamlit_App/app.py
```

The app loads `Models/titanic_model.pkl` relative to `app.py`, so run it from the repository root as shown above.

## Deployment

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud). To deploy your own copy, push this repo to GitHub and point Streamlit Community Cloud at `Streamlit_App/app.py`.

## Requirements

See [requirements.txt](requirements.txt): `streamlit`, `pandas`, `numpy`, `scikit-learn==1.9.1` (the model pickle must match this scikit-learn version).
