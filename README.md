# Multimodal Clustering — Amazon Fine Food Reviews

---

## Project Structure

```
amazon_review_clustering/
├── main.py                      ← run everything from here
├── requirements.txt
├── README.md
├── data/
│   └── Reviews.csv              ← place your dataset here
├── src/
│   ├── config.py                ← all settings in one place
│   ├── preprocessing.py         ← load, clean, TF-IDF, SVD, fusion, PCA
│   ├── eda.py                   ← all EDA charts (cells 7,8,9,16)
│   ├── clustering.py            ← K-Means, DBSCAN (k-dist tuned), Agglomerative, Louvain
│   ├── results.py               ← t-SNE, profiles, word clouds, top terms, dashboard
│   ├── temporal.py              ← [NEW] temporal drift + seasonal analysis
│   ├── category_metadata.py     ← [NEW] product category enrichment
│   ├── topic_modeling.py        ← [NEW] LDA sub-topics per cluster
│   └── llm_labeling.py          ← [NEW] Claude API auto-labels clusters
├── outputs/                     ← all charts + CSV saved here
└── reports/
    └── capstone_report.md       ← fill in your results here
```

---

## Quickstart

```cmd
pip install -r requirements.txt
python main.py --skip-llm        # no API key needed
python main.py                   # full pipeline with LLM labels
python main.py --samples 50000   # match original notebook (needs more RAM/time)
python main.py --samples 1000 --skip-llm  # quick test run
```

---

## What Each Run Produces

| Output File | From |
|---|---|
| `eda_overview.png` | Rating dist, word count, helpfulness, reviews/year |
| `eda_text_insights.png` | Word cloud, 5★ vs 1★ over time, helpfulness scatter |
| `num_correlation.png` | Numerical feature correlation heatmap |
| `pca_analysis.png` | Scree plot + cumulative variance |
| `elbow_silhouette.png` | K-Means optimal k selection |
| `dbscan_kdistance.png` | **[NEW]** k-distance plot for eps auto-tuning |
| `network_graph_clustering.png` | Louvain community detection |
| `tsne_visualization.png` | t-SNE: K-Means / rating / review length |
| `cluster_profiles.png` | Heatmap + stacked rating bars |
| `cluster_wordclouds.png` | Word cloud per cluster |
| `algorithm_comparison.png` | K-Means vs DBSCAN vs Agglomerative |
| `final_dashboard.png` | Full summary dashboard |
| `temporal_clustering.png` | **[NEW]** Cluster drift over time |
| `seasonal_complaints.png` | **[NEW]** Holiday complaint spike analysis |
| `category_cluster_analysis.png` | **[NEW]** Product category × cluster heatmap |
| `topic_modeling_per_cluster.png` | **[NEW]** LDA sub-topics per cluster |
| `llm_cluster_labels.png` | **[NEW]** LLM auto-generated cluster cards |
| `amazon_reviews_clustered.csv` | Full dataset with all cluster columns |

---

## Five Advanced Features Added

| # | Feature | Module |
|---|---|---|
| 1 | Product Category Metadata | `src/category_metadata.py` |
| 2 | Temporal Clustering Analysis | `src/temporal.py` |
| 3 | Optimal DBSCAN Tuning (k-distance plot) | `src/clustering.py` |
| 4 | Topic Modeling per Cluster (LDA) | `src/topic_modeling.py` |
| 5 | LLM Auto-Labeling (Claude API) | `src/llm_labeling.py` |

---
