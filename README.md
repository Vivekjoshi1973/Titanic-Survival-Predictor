# Titanic Survival Predictor

A Streamlit app that predicts whether a Titanic passenger survived, using a Logistic Regression model trained on the classic Titanic dataset.

**Live app:** https://titanic-survival-predictor-vj.streamlit.app/

## Features

- Ticket class, sex, age, siblings/spouses, parents/children, fare, port of embarkation, and travelling-alone inputs
- One-click survival prediction
- Model cached with `st.cache_resource` for fast load times

## Project structure

```
Titanic Survival Predictor/
├── Models/
│   └── titanic_model.pkl      # trained Logistic Regression model
├── Notebook/
│   └── titanic.ipynb          # EDA and model training
├── Streamlit_App/
│   └── app.py                 # Streamlit app
├── requirements.txt
└── README.md
```

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
