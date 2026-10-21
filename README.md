# Ransomware Detection - Web Application

Flask backend + HTML/CSS/JavaScript frontend for the project
"Ransomware Detection Methods: A Comprehensive Survey and Analysis".
It uses your original pipeline (TF-IDF + SelectKBest + AdaBoost + DNN) and your dataset `data_file.csv`.

## Run it

```bash
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000, create an account, sign in.

The `models/` folder already contains trained models. If they cannot be loaded
(for example a different scikit-learn version), open **Train models** in the app
or run `python train_model.py` (about 40 seconds).

Optional: `pip install tensorflow` and the neural network is trained with Keras
(same layers as your original script). Without it, scikit-learn's MLPClassifier is used.

## Pages

| Page | What it does |
|---|---|
| Register / Sign in | Accounts in `users.db` (same `users` table as before) |
| Overview | Progress through the pipeline, dataset facts, latest results |
| Dataset | Preview, columns and missing values, statistics, class balance; upload your own CSV |
| Train models | Choose label column, number of features (k), test size, epochs; live progress |
| Check a file | Enter header values for one file, or upload a CSV for many files; verdict and confidence |
| Results | Accuracy, precision, recall, F1, FPR for both models; chart; confusion matrices; feature importance |

## Files

```
app.py            Flask routes (pages + JSON API), login, CSRF protection
ml_pipeline.py    Preprocessing, training, saving/loading, prediction
train_model.py    Train from the command line
data/             data_file.csv (your dataset); uploaded CSVs are saved here
models/           bundle.joblib + metrics.json (created by training)
templates/        HTML pages (Jinja2)
static/           CSS and JavaScript (no external JS libraries needed)
```

## What changed compared with the Streamlit version

* Prediction is now real. In `app2.py` the Predict page returned fixed placeholder values.
  The fitted TF-IDF vectorisers and feature selector are now saved with the model,
  so any file can be classified.
* Original script results reproduced exactly: AdaBoost accuracy 99.96%, precision 99.98%,
  recall 99.93%, F1 99.95%, FPR 0.014%.
* Passwords use salted hashes. Old SHA-256 accounts still work and are upgraded on first login.
* Training runs in the background with a progress display.

## Things to know (good for your viva)

1. **Neural network result differs from the PPT.** Your PPT shows DNN accuracy 81.7%. That model was
   trained on raw values (some columns reach billions). This app standardises the inputs for the
   neural network by default, which gives about 99.97%. Untick
   "Standardise features" on the Train page to reproduce the unscaled setup.
2. **File names matter to the model.** 7 of the 20 selected features are words from `FileName`
   (exe, dll, system, ...). Using only the 13 numeric header fields, AdaBoost reaches about 97.8%.
   In this dataset the malicious files are named like `VirusShare_<hash>`, so the model can partly
   learn the naming of the dataset rather than ransomware behaviour. Mention this as a limitation.
3. **Feature selection is fitted on the full dataset before the train/test split**, as in the original
   script. A stricter evaluation would fit TF-IDF and SelectKBest on the training part only.
4. In this dataset `Benign = 1`. A "false positive" is therefore a malicious file called benign.
5. Results are for this dataset only; they do not guarantee detection of new ransomware.

## Notes for deployment

`python app.py` uses Flask's development server on localhost, which is fine for a demo.
For a real deployment use a production server (for example gunicorn) behind HTTPS and set the
`SECRET_KEY` environment variable.
