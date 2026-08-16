 # Book Rating Prediction

Predict a book's Goodreads-style `average_rating` from metadata alone (pages, reviews, publisher, language, and simple engineered features).

## Problem statement

Book ratings are subjective, but publishers and platforms often only have structured metadata at hand. This project builds a regression pipeline that estimates `average_rating` from those fields, compares several models, and serves the winning model through a small Streamlit app. The goal is an honest, reproducible baseline — not a claim that metadata can fully explain taste.

## Dataset

- **File:** `books.csv` (Goodreads-style book catalog)
- **Approx. size:** ~11,127 raw rows; ~11,123 after skipping malformed CSV lines; fewer after cleaning (bad dates and zero ratings removed)
- **Target:** `average_rating`
- **Raw columns:** `bookID`, `title`, `authors`, `average_rating`, `isbn`, `isbn13`, `language_code`, `num_pages`, `ratings_count`, `text_reviews_count`, `publication_date`, `publisher`

## Project structure

```
book_rating_prediction_project/
├── README.md
├── requirements.txt
├── books.csv
├── book_rating_prediction.ipynb   # EDA → cleaning → features → models → save
├── app.py                         # Streamlit inference UI
└── models/
    └── book_rating_bundle.joblib  # model + encoders + median + feature order
```

## How to run

### 1. Get the project

Download/unzip this archive (or clone the repo) and open a terminal in the project folder.

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
# Optional, for the notebook UI:
pip install jupyter
```

### 4. Run the notebook (training / learning path)

```bash
jupyter notebook book_rating_prediction.ipynb
```

Run all cells top to bottom. Step 7 writes `models/book_rating_bundle.joblib`. A pre-trained bundle is already included so the app works without re-training.

### 5. Launch the Streamlit app

```bash
python3 -m streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`).

**Sample input (defaults in the form):**

| Field | Example |
|---|---|
| Title | Harry Potter and the Half-Blood Prince |
| Authors | J.K. Rowling/Mary GrandPré |
| Language code | eng |
| Pages | 652 |
| Ratings count | 10000 |
| Text reviews count | 500 |
| Publisher | Scholastic Inc. |
| Publication date | 9/16/2006 |

**Expected output:** a predicted average rating on a ~1–5 scale (e.g. about **4.0** for the sample above), plus an expandable view of the engineered feature row. Unknown publishers/languages show a warning and use a training-known fallback.

## Results summary

On an 80/20 train/test split (`random_state=42`):

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Gradient Boosting** | 0.1983 | 0.2689 | **0.1504** |
| Random Forest | 0.1981 | 0.2705 | 0.1398 |
| Linear Regression | 0.2122 | 0.2831 | 0.0580 |
| Decision Tree | 0.2819 | 0.3790 | −0.6878 |

**Gradient Boosting** was selected (best R², essentially tied with Random Forest on MAE). The single decision tree overfits badly (negative R²).

## Known limitations

- **Low R² (~0.15) is expected.** Ratings reflect subjective content quality; page count, publisher ID, and review volume are weak proxies. A low-but-nonzero R² is an honest finding, not a failed model.
- **Unseen categories.** Publishers/languages not seen in training cannot be label-encoded as-is. The app maps them to a training-mode fallback (`eng` / `Vintage`) and warns the user — so inference continues with a defined (but approximate) path instead of crashing.
- **Label encoding + trees.** Encoders assume the same category→integer mapping as training; the saved bundle must be used for inference.
