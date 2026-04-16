# src/config.py — Global constants for the entire pipeline
import os
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data"
OUTPUT_DIR  = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

DATA_PATH    = DATA_DIR / "Reviews.csv"
RANDOM_STATE = 42
N_SAMPLES    = 50_000      # bump to 50_000 if you have time/RAM
OPTIMAL_K    = 10

# Text
TFIDF_MAX_FEATURES = 8_000 
TFIDF_NGRAM_RANGE  = (1, 2)
TFIDF_MIN_DF       = 5
TFIDF_MAX_DF       = 0.85
N_SVD              = 100
N_PCA              = 30

# Fusion weights
TEXT_WEIGHT    = 1.0
NUMERIC_WEIGHT = 0.6

# DBSCAN
DBSCAN_MIN_SAMPLES = 20

# Graph
K_NEIGHBORS = 10

# Topic modeling
LDA_N_TOPICS = 4
LDA_MAX_ITER = 15

# LLM
LLM_MODEL        = "claude-sonnet-4-20250514"
LLM_MAX_TOKENS   = 200
LLM_TOP_TERMS    = 12
LLM_N_REVIEWS    = 3

# Colors — from your original notebook
CLUSTER_COLS = ['#e53935', '#1e88e5', '#43a047', '#fb8c00', '#8e24aa']
RATING_COLS  = ['#d32f2f', '#f57c00', '#fbc02d', '#388e3c', '#1976d2']
