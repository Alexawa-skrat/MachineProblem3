# LMS Student Performance Predictor (Machine Problem 3)

Streamlit app that predicts whether a student's midterm LMS activity maps to a **High** or **Low** rating.

## Structure
- `features.py` – shared feature engineering (log1p, `_plus_`, `_times_`; 72 features)
- `train_model.py` – trains the soft voting ensemble, saves `artifacts/`
- `app.py` – Streamlit web app
- `requirements.txt`
- `data/` – put `augmented_balanced_dataset_m.csv` here
- `artifacts/` – generated: `model.joblib`, `metrics.json`, `stats.json`

## Run locally
1. `python -m venv .venv` then activate it
2. `pip install -r requirements.txt`
3. Copy `augmented_balanced_dataset_m.csv` into `data/`
4. `python train_model.py`
5. `streamlit run app.py`

## Deploy (Streamlit Community Cloud)
1. Push this folder to a public GitHub repo, including `artifacts/` (so the app loads instantly).
2. Go to share.streamlit.io, sign in with GitHub, click **New app**.
3. Pick the repo, branch `main`, main file `app.py`, then **Deploy**.
4. Submit the generated `*.streamlit.app` URL.
