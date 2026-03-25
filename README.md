# Hotel Review Satisfaction Classification
### TF-IDF + Logistic Regression vs Fine-Tuned DistilBERT

> **University Assignment — Using Text as Data**

---

## Project Overview

This project investigates whether hotel guest satisfaction ratings can be automatically inferred from free-text reviews, and whether contextual transfer learning via DistilBERT provides a practically meaningful improvement over a tuned TF-IDF Logistic Regression baseline.

**Research question:**
> "To what extent can automated text classification models accurately infer hotel guest satisfaction ratings from review text, and does fine-tuned DistilBERT provide a practically meaningful improvement over a tuned TF-IDF Logistic Regression baseline for this five-class prediction task?"

**Task type:** Five-class text classification (star ratings 1–5)

**Primary evaluation metric:** Macro F1 (chosen due to class imbalance; treats all rating tiers equally)

---

## Dataset

| Property | Value |
|---|---|
| **Name** | TripAdvisor Hotel Reviews |
| **Source** | [Kaggle — andrewmvd](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews) |
| **File** | `tripadvisor_hotel_reviews.csv` |
| **Key columns** | `Review` (free text), `Rating` (integer 1–5) |
| **Language** | English |
| **License** | CC0: Public Domain |

> **Note:** The dataset file is NOT included in this repository due to its size. Please download it directly from Kaggle (free account required) and place it in the project root directory, or update the `DATA_PATH` variable in the script.

---

## Repository Structure

```
hotel_review_classification/
│
├── hotel_review_classification_run_final.ipynb # Notebook with full run outputs
├── README.md                                   # This file
│
├── outputs/                         # Auto-created when script runs
│   ├── fig1_class_distribution.png
│   ├── fig2_length_distribution.png
│   ├── fig3_cm_lr.png
│   ├── fig4_cm_bert.png
│   ├── fig5_cm_comparison.png
│   ├── fig6_per_class_f1.png
│   ├── model_comparison.csv
│   └── final_model_selection.txt
│
└── tripadvisor_hotel_reviews.csv    # ← Download from Kaggle (not in repo)
```

---

## Installation

### Option A — Google Colab (recommended)

1. Clone or download this repository.
2. Upload `hotel_review_classification_run_final.ipynb` to Google Colab.
3. Enable GPU: **Runtime → Change runtime type → T4 GPU**
4. Upload `tripadvisor_hotel_reviews.csv` to the Colab session (Files panel on the left).
5. Run all cells from top to bottom — Section 0 installs all required packages automatically.

### Option B — Local environment

```bash
# 1. Clone the repository
git clone https://github.com/Chanyah/hotel_review_classification.git
cd hotel_review_classification

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate

# 3. Install dependencies (no version pins — use latest compatible versions)
pip install scikit-learn transformers datasets accelerate pandas matplotlib seaborn torch
```

> ⚠️ **Note on versions:** Do not pin `transformers` to a specific version. The code requires `transformers>=4.41.0` for the `eval_strategy` parameter and compatible `accelerate`. Installing without pins picks the correct versions automatically.

---

## How to Run

### Step 1 — Download the data

1. Go to https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews
2. Download `tripadvisor_hotel_reviews.csv`
3. Place the file in the project root directory (same folder as the script)

### Step 2 — Run the notebook

Open `hotel_review_classification_run_final.ipynb` in Google Colab and run all cells from top to bottom.

Update the `DATA_PATH` constant in Section 1 if your CSV is in a different location.

### Step 3 — Inspect outputs

All figures and metric files are saved automatically to the `./outputs/` directory. The final model selection and justification are printed to the console and saved to `outputs/final_model_selection.txt`.

---

## Expected Outputs

After a successful run, the `outputs/` directory will contain:

| File | Description |
|---|---|
| `fig1_class_distribution.png` | Bar charts of class frequency and percentage |
| `fig2_length_distribution.png` | Review character and word count histograms |
| `fig3_cm_lr.png` | Confusion matrix — TF-IDF + LR (tuned) |
| `fig4_cm_bert.png` | Confusion matrix — DistilBERT (tuned) |
| `fig5_cm_comparison.png` | Side-by-side confusion matrix comparison |
| `fig6_per_class_f1.png` | Per-class F1 bar chart: both models |
| `lr_gridsearch_results.csv` | Full GridSearchCV result table (all parameter combinations) |
| `model_comparison.csv` | Summary table: accuracy, macro F1, weighted F1 for both models |
| `final_model_selection.txt` | Final model choice and written justification |
| `class_distribution.csv` | Class counts and percentages |

Console output includes full `classification_report` for both models, with per-class precision, recall, and F1.

---

## Approximate Runtime

| Component | CPU | Colab T4 GPU |
|---|---|---|
| Data loading + EDA | ~30 sec | ~30 sec |
| GridSearchCV (TF-IDF + LR) | ~5–15 min | ~5–15 min |
| DistilBERT training (×2 LR, 3 epochs) | **~4–8 hours** | ~25–40 min |
| **Total** | **Very slow — GPU strongly recommended** | **~45–60 min** |

> DistilBERT requires a GPU for practical use. On free Colab (T4), the full training run takes approximately 45–60 minutes.

---

## Reproducibility Notes

- **Random seed `42`** is fixed at the start of every section (`random`, `numpy`, `torch.cuda`).
- **Stratified splitting** preserves the original class distribution in train and test sets.
- **No test-set leakage**: the TF-IDF vocabulary is fitted inside a scikit-learn `Pipeline` on training data only; DistilBERT tokenisation is applied after splitting.
- **Package versions** are pinned in `requirements.txt` to ensure consistency.
- Results may differ by ±0.001 across runs on different GPU hardware due to non-deterministic CUDA operations; this is negligible for comparison purposes.
- All numerical results reported in the paper are generated by running this script in full without modification.

---

## Limitations

- DistilBERT hyperparameter search is bounded to two learning rates (2e-5, 3e-5) due to Colab GPU constraints; a broader search may yield different results.
- Reviews longer than 128 tokens are truncated; tail content (often containing summary judgements) is discarded for DistilBERT.
- Dataset covers only English TripAdvisor reviews; results may not generalise to other platforms or languages.
- Class imbalance (4- and 5-star reviews dominate) means minority-class metrics carry higher variance.

---

## Contact / Author

- **Name:** Chanya Hirunpruk
- **Institution:** Univerity of Exeter
- **Module:** SSIM916 Machine Learning
- **Submission date:** 25 March 2026
- **GitHub:** https://github.com/Chanyah/hotel_review_classification
