<div align="center">

# Credit Card Fraud Detection Web App

**A small Flask app that classifies a credit card transaction as fraudulent or legitimate with a pre-trained logistic regression model.**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Heroku](https://img.shields.io/badge/Procfile-Heroku--style-430098?logo=heroku&logoColor=white)

</div>

## Overview

The app loads a pickled scikit-learn `LogisticRegression` (`model.pkl`, trained with `class_weight="balanced"`) and exposes it through a Flask web form. A user pastes the numeric values of one transaction, and the app returns the model's prediction (the label convention follows the sample files: `1` = fraud, `0` = valid).

## How it works

- `GET /` renders the home page (`home.html`).
- `POST /predict` reads the form field `message`, splits it on whitespace, converts each token to a float, reshapes it into one row and calls `clf.predict`. The result is rendered in `result.html`.

## Sample data

Two CSVs are included to try the app. Columns are `Time`, `V1`-`V28`, `Amount`, `class`; the values appear already scaled/anonymized, in the style of the Kaggle credit card fraud data.

| File | Rows | `class` |
|------|------|---------|
| `fraud_values.csv` | 24 | 1 (fraud) |
| `valid_values.csv` | 25 | 0 (valid) |

To test, copy one row **without** the final `class` column and paste its values, separated by spaces, into the form.

## Tech Stack

Python, Flask, pandas, NumPy, scikit-learn (for the pickled model), Gunicorn (via `Procfile`).

## Getting Started

The repository has no `requirements.txt`, so install the dependencies directly:

```bash
pip install flask pandas numpy scikit-learn gunicorn
```

Before running, note two things about the current code:

1. `app.py` loads the model from a hard-coded Windows path. Change `filename` to a relative path such as `model.pkl`.
2. The templates `home.html` and `result.html` are expected in a `templates/` folder but are not included in this repository; add them before running.

Then start the app:

```bash
python app.py                # development server, http://127.0.0.1:5000
gunicorn app:app             # production-style, as defined in the Procfile
```

> Only load pickle files you trust, and use a scikit-learn version close to the one that produced `model.pkl`.

## Project Structure

```
├── app.py               # Flask application
├── model.pkl            # Pickled logistic regression model
├── fraud_values.csv     # Sample fraudulent transactions
├── valid_values.csv     # Sample legitimate transactions
└── Procfile             # web: gunicorn app:app
```
