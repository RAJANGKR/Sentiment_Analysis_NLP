# Movie Review Sentiment Analysis

A Natural Language Processing (NLP) project that classifies IMDb movie reviews as **Positive** or **Negative** using text preprocessing, TF-IDF vectorization, and Logistic Regression.

## Overview

This project performs binary sentiment classification on a dataset of 50,000 movie reviews. The pipeline covers full text cleaning, feature extraction using TF-IDF, and a machine learning classifier that predicts sentiment with **~90% accuracy**.

## Dataset

- **Size:** 50,000 movie reviews
- **Columns:** `review` (raw text), `sentiment` (positive/negative)
- **Source:** IMDb-style movie review dataset

## Tech Stack

- **Language:** Python
- **Libraries:** pandas, NumPy, NLTK, BeautifulSoup, contractions, scikit-learn, matplotlib, seaborn
- **Environment:** Google Colab / Jupyter Notebook

## Project Pipeline

```
Raw Reviews
   ↓
Text Preprocessing (HTML removal, lowercasing, contraction expansion,
                     URL removal, punctuation cleanup, stopword removal,
                     lemmatization)
   ↓
Train-Test Split (80/20, stratified)
   ↓
TF-IDF Vectorization (unigrams + bigrams, top 5000 features)
   ↓
Logistic Regression (classification model)
   ↓
Evaluation (accuracy, precision, recall, F1-score, confusion matrix)
```

## Text Preprocessing Steps

| Step | Purpose |
|------|---------|
| HTML tag removal | Removes leftover tags like `<br />` |
| Lowercasing | Normalizes word casing |
| Contraction expansion | `don't` → `do not` (preserves negation meaning) |
| URL removal | Removes irrelevant links |
| Repeated character normalization | `goooood` → `good` |
| Special character cleanup | Keeps letters and key punctuation (`. ! ?`) |
| Stopword removal | Removes filler words while **keeping negations** (`not`, `no`, `never`) |
| Lemmatization | Reduces words to base form (`loved` → `love`) |

## Model Details

- **Feature extraction:** `TfidfVectorizer` (`max_features=5000`, `ngram_range=(1,2)`)
- **Classifier:** `LogisticRegression` (scikit-learn)
- **Why Logistic Regression:** Despite the name, it is a classification algorithm. It uses a sigmoid function to output a probability (0–1), which is then thresholded at 0.5 to assign a Positive/Negative label.

## Results

| Metric | Score |
|--------|-------|
| Accuracy | ~90% |
| Precision (avg) | 0.90 |
| Recall (avg) | 0.90 |
| F1-score (avg) | 0.90 |

Confusion matrix and full classification report are generated in the notebook.

## Known Limitation

TF-IDF + Logistic Regression treats text as independent word/phrase weights and does not fully capture sentence-level context. Reviews with subtle or sarcastic negation (e.g., *"not very unique"*) can occasionally be misclassified. Potential improvements include trigram features, explicit negation tagging, or transformer-based models (e.g., BERT).

## How to Run

1. Open the notebook in Google Colab or Jupyter.
2. Install dependencies:
   ```
   pip install nltk beautifulsoup4 contractions scikit-learn pandas numpy matplotlib seaborn
   ```
3. Run the preprocessing block to clean the review text.
4. Run the training block to vectorize text and train the model.
5. Use the `predict_sentiment()` function to test custom reviews.

## Example Prediction

```python
predict_sentiment("This movie was absolutely fantastic, great acting and story!")
# Output: Positive (97.6% confidence)
```

## Future Improvements

- Add trigrams and tune `min_df` / `max_df` for better negation handling
- Compare with Naive Bayes and SVM classifiers
- Experiment with word embeddings (Word2Vec, GloVe)
- Fine-tune a transformer model (BERT/DistilBERT) for higher accuracy

## Author

**Rajan Garudkar**


