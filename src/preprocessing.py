# src/preprocessing.py
# Cells 5, 11, 12, 13, 14 from your notebook — load, clean, featurize, fuse
import re, ssl, os, pathlib
import numpy as np
import pandas as pd
import nltk

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler

from src.config import (
    DATA_PATH, N_SAMPLES, RANDOM_STATE,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MIN_DF, TFIDF_MAX_DF,
    N_SVD, N_PCA, TEXT_WEIGHT, NUMERIC_WEIGHT
)

STOP_WORDS_SET = set(stopwords.words('english'))


def get_data_path(override=None):
    """Find Reviews.csv — checks data/ folder first, then common cache paths."""
    if override:
        return override
    candidates = [
        str(DATA_PATH),
        'Reviews.csv',
        '/root/.cache/kagglehub/datasets/snap/amazon-fine-food-reviews/versions/2/Reviews.csv',
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Reviews.csv not found. Place it in the data/ folder."
    )


# ── Cell 5: Load & Clean ───────────────────────────────────────────────────
def load_data(path=None, n_samples=N_SAMPLES):
    path = get_data_path(path)
    print('Loading dataset...')
    df_full = pd.read_csv(path)
    print(f'  Raw dataset : {df_full.shape[0]:,} rows × {df_full.shape[1]} columns')

    df_full.drop_duplicates(subset=['UserId', 'Time', 'Text'], inplace=True)
    df_full.dropna(subset=['Text', 'Summary'], inplace=True)
    print(f'  After dedup : {df_full.shape[0]:,} rows')

    df = df_full.sample(n=min(n_samples, len(df_full)), random_state=RANDOM_STATE).reset_index(drop=True)

    df['Date']           = pd.to_datetime(df['Time'], unit='s')
    df['Year']           = df['Date'].dt.year
    df['Month']          = df['Date'].dt.month
    df['Quarter']        = df['Date'].dt.quarter
    df['text_length']    = df['Text'].str.len()
    df['word_count']     = df['Text'].str.split().str.len()
    df['summary_length'] = df['Summary'].str.len()
    df['helpfulness_ratio'] = df['HelpfulnessNumerator'] / (df['HelpfulnessDenominator'] + 1)

    print(f'  Working sample : {len(df):,} reviews')
    print(f'  Date range     : {df["Date"].min().date()} → {df["Date"].max().date()}')
    print(f'  Unique products: {df["ProductId"].nunique():,}')
    print(f'  Unique users   : {df["UserId"].nunique():,}')
    return df


# ── Cell 11: Text Preprocessing ───────────────────────────────────────────
def preprocess(text):
    """Lowercase, strip HTML/punctuation, normalize whitespace."""
    text = str(text).lower()
    text = re.sub(r'<[^>]+>',   ' ', text)
    text = re.sub(r'http\S+',   ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def add_clean_text(df):
    print('Preprocessing text (Summary + Review body)...')
    df['clean_text'] = (df['Summary'].fillna('') + ' ' + df['Text']).apply(preprocess)
    print(f'  Avg clean text length: {df["clean_text"].str.len().mean():.0f} chars')
    print(f'  Example: {df["clean_text"].iloc[0][:150]}...')
    return df


# ── Cell 12: TF-IDF → TruncatedSVD (LSA) ─────────────────────────────────
def build_text_features(df):
    print('Step 1 — TF-IDF vectorization...')
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF, max_df=TFIDF_MAX_DF,
        sublinear_tf=True, stop_words='english'
    )
    X_tfidf = tfidf.fit_transform(df['clean_text'])
    print(f'  TF-IDF matrix : {X_tfidf.shape}')

    print(f'Step 2 — TruncatedSVD ({N_SVD} components)...')
    svd    = TruncatedSVD(n_components=N_SVD, random_state=RANDOM_STATE)
    X_text = svd.fit_transform(X_tfidf)
    print(f'  SVD shape     : {X_text.shape}')
    print(f'  Var explained : {svd.explained_variance_ratio_.sum():.2%}')
    return X_text, tfidf, svd


# ── Cell 13: Numerical Features ───────────────────────────────────────────
def build_numeric_features(df):
    print('Step 3 — Numerical features...')
    df['log_word_count']     = np.log1p(df['word_count'])
    df['log_summary_length'] = np.log1p(df['summary_length'])
    df['log_total_votes']    = np.log1p(df['HelpfulnessDenominator'])
    df['rating_norm']        = (df['Score'] - 1) / 4

    use_cols = ['rating_norm', 'helpfulness_ratio',
                'log_word_count', 'log_summary_length', 'log_total_votes']
    scaler = StandardScaler()
    X_num  = scaler.fit_transform(df[use_cols])
    print(f'  Numerical features: {X_num.shape}  cols={use_cols}')
    return X_num, scaler, use_cols


# ── Cell 14: Multimodal Fusion → PCA ──────────────────────────────────────
def fuse_and_reduce(X_text, X_num):
    X_fused = np.hstack([X_text * TEXT_WEIGHT, X_num * NUMERIC_WEIGHT])
    print(f'Fused matrix shape: {X_fused.shape}')

    pca   = PCA(n_components=N_PCA, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_fused)
    print(f'PCA output shape  : {X_pca.shape}')
    print(f'Variance explained: {pca.explained_variance_ratio_.sum():.2%}')
    return X_pca, X_fused, pca


def run_preprocessing(path=None, n_samples=N_SAMPLES):
    df = load_data(path, n_samples)
    df = add_clean_text(df)
    X_text, tfidf, svd   = build_text_features(df)
    X_num, scaler, cols  = build_numeric_features(df)
    X_pca, X_fused, pca  = fuse_and_reduce(X_text, X_num)
    arts = dict(tfidf=tfidf, svd=svd, scaler=scaler, pca=pca,
                X_text=X_text, X_num=X_num, X_fused=X_fused)
    print('\n✅ Preprocessing complete.')
    return df, X_pca, arts
