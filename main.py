"""
main.py — End-to-End Multimodal Clustering Pipeline
Amazon Fine Food Reviews | Capstone Project

Usage:
  python main.py                   # full pipeline (5,000 reviews)
  python main.py --samples 50000   # match original notebook (50k)
  python main.py --skip-llm        # skip LLM labeling (no API key needed)
  python main.py --k 4             # override cluster count
"""
import argparse
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # change to 'TkAgg' or remove if running locally
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
warnings.filterwarnings('ignore')

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# ── src modules ───────────────────────────────────────────────────────────
from src.config import DATA_PATH, N_SAMPLES, OPTIMAL_K, RANDOM_STATE, OUTPUT_DIR
from src.preprocessing   import run_preprocessing
from src.eda             import (print_summary_stats, plot_eda_overview,
                                  plot_eda_text_insights, plot_numeric_correlation,
                                  plot_pca_analysis)
from src.clustering      import (find_optimal_k, run_kmeans, run_dbscan,
                                  run_agglomerative, run_louvain,
                                  plot_network_graph, print_algorithm_comparison)
from src.results         import (plot_tsne_results, plot_cluster_profiles,
                                  plot_wordclouds, print_top_terms,
                                  plot_algorithm_comparison, print_sample_reviews,
                                  plot_final_dashboard)
# Advanced features
from src.category_metadata import enrich_with_categories, cross_category_cluster_analysis
from src.temporal          import plot_temporal_drift, seasonal_complaint_analysis
from src.topic_modeling    import run_lda_per_cluster, plot_topic_heatmap, print_topic_summary
from src.llm_labeling      import auto_label_clusters, plot_labeled_clusters


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--samples',  type=int,  default=N_SAMPLES)
    p.add_argument('--k',        type=int,  default=None)
    p.add_argument('--skip-llm', action='store_true')
    p.add_argument('--data',     type=str,  default=str(DATA_PATH))
    return p.parse_args()


def run_tsne(X_fused, n_pre=20):
    print('\nRunning t-SNE (pre-reduced to 20D with PCA)...')
    pca20 = PCA(n_components=min(n_pre, X_fused.shape[1]), random_state=RANDOM_STATE)
    X20   = pca20.fit_transform(X_fused)
    tsne  = TSNE(n_components=2, perplexity=35, learning_rate='auto',
                 init='pca', max_iter=750, random_state=RANDOM_STATE)
    result = tsne.fit_transform(X20)
    print('t-SNE complete!')
    return result


def main():
    args    = parse_args()
    t_start = time.time()

    print('=' * 70)
    print('  MULTIMODAL CLUSTERING — AMAZON FINE FOOD REVIEWS')
    print('  Capstone Project | End-to-End Pipeline')
    print('=' * 70)

    # ════════════════════════════════════════════════════════════════════
    # SECTION i-ii: Introduction + Data Loading
    # ════════════════════════════════════════════════════════════════════
    print('\n[1/9] Loading & preprocessing data...')
    df, X_pca, arts = run_preprocessing(path=args.data, n_samples=args.samples)

    # ════════════════════════════════════════════════════════════════════
    # SECTION iii-a: Basic EDA — Distributions (Cells 7, 8, 9)
    # ════════════════════════════════════════════════════════════════════
    print('\n[2/9] Exploratory Data Analysis...')
    print_summary_stats(df)
    plot_eda_overview(df)
    plot_eda_text_insights(df)
    plot_numeric_correlation(arts['X_num'],
                             ['rating_norm','helpfulness_ratio',
                              'log_word_count','log_summary_length','log_total_votes'])

    # ════════════════════════════════════════════════════════════════════
    # SECTION iii-b/c: Feature Engineering + PCA (Cells 11–16)
    # ════════════════════════════════════════════════════════════════════
    print('\n[3/9] PCA analysis...')
    plot_pca_analysis(arts['pca'])

    # ════════════════════════════════════════════════════════════════════
    # SECTION iii-d: Clustering (Cells 18–22)
    # ════════════════════════════════════════════════════════════════════
    print('\n[4/9] Clustering...')

    # K-Means: find optimal k
    best_k, inertias, sil_scores = find_optimal_k(X_pca)
    optimal_k = args.k if args.k else best_k

    # Final K-Means
    df['cluster'], km, final_sil = run_kmeans(X_pca, k=optimal_k)

    # t-SNE for visualization
    X_tsne       = run_tsne(arts['X_fused'])
    df['tsne_x'] = X_tsne[:, 0]
    df['tsne_y'] = X_tsne[:, 1]

    # DBSCAN with k-distance auto-tuned eps (Feature #3)
    df['dbscan_cluster'], db, n_db_clusters, n_noise = run_dbscan(X_tsne)

    # Agglomerative
    df['agg_cluster'], agg, agg_sil, X_pca10 = run_agglomerative(arts['X_fused'], k=optimal_k)

    # ════════════════════════════════════════════════════════════════════
    # SECTION iii-e: Graph / Louvain Clustering (Cells 23–27)
    # ════════════════════════════════════════════════════════════════════
    print('\n[5/9] Graph clustering (Louvain)...')
    df['graph_cluster'], partition, n_comm, modularity, graph_sil = run_louvain(X_pca)
    plot_network_graph(df, n_comm, n_clusters=optimal_k, final_sil=final_sil)
    print_algorithm_comparison(df, final_sil, agg_sil, graph_sil,
                               n_comm, modularity, n_clusters=optimal_k)

    # ════════════════════════════════════════════════════════════════════
    # SECTION iv: Results (Cells 29–35)
    # ════════════════════════════════════════════════════════════════════
    print('\n[6/9] Results & visualizations...')
    plot_tsne_results(df, optimal_k)
    profile, rating_pct = plot_cluster_profiles(df, optimal_k)
    plot_wordclouds(df, optimal_k)
    print_top_terms(df, optimal_k)
    plot_algorithm_comparison(df, final_sil, agg_sil, n_db_clusters, n_noise, optimal_k)
    print_sample_reviews(df, optimal_k)
    plot_final_dashboard(df, profile, rating_pct, arts['pca'], final_sil, agg_sil, optimal_k)

    # ════════════════════════════════════════════════════════════════════
    # ADVANCED FEATURE #1: Product Category Metadata
    # ════════════════════════════════════════════════════════════════════
    print('\n[7/9] Advanced Feature #1 — Category Metadata...')
    df = enrich_with_categories(df)
    cross_category_cluster_analysis(df)

    # ════════════════════════════════════════════════════════════════════
    # ADVANCED FEATURE #2: Temporal Clustering
    # ════════════════════════════════════════════════════════════════════
    print('\nAdvanced Feature #2 — Temporal Analysis...')
    plot_temporal_drift(df)
    seasonal_complaint_analysis(df)

    # ════════════════════════════════════════════════════════════════════
    # ADVANCED FEATURE #4: Topic Modeling per Cluster (LDA)
    # ════════════════════════════════════════════════════════════════════
    print('\n[8/9] Advanced Feature #4 — Topic Modeling per Cluster (LDA)...')
    all_topics = run_lda_per_cluster(df)
    plot_topic_heatmap(all_topics)
    print_topic_summary(all_topics)

    # ════════════════════════════════════════════════════════════════════
    # ADVANCED FEATURE #5: LLM Auto-Labeling
    # ════════════════════════════════════════════════════════════════════
    print('\n[9/9] Advanced Feature #5 — LLM Cluster Labeling...')
    if not args.skip_llm:
        labels = auto_label_clusters(df)
        plot_labeled_clusters(df, labels)
    else:
        print('  Skipped (--skip-llm flag set). Remove flag to enable.')
        labels = {c: {'label': f'Cluster {c}', 'description': '', 'sentiment': ''}
                  for c in df['cluster'].unique()}

    # ════════════════════════════════════════════════════════════════════
    # Save final CSV
    # ════════════════════════════════════════════════════════════════════
    out_csv = OUTPUT_DIR / 'amazon_reviews_clustered.csv'
    df.to_csv(out_csv, index=False)

    elapsed = time.time() - t_start
    print('\n' + '=' * 70)
    print(f'✅  Pipeline complete in {elapsed/60:.1f} minutes.')
    print(f'    Outputs saved to: {OUTPUT_DIR}/')
    print('\n  Files generated:')
    for f in sorted(OUTPUT_DIR.glob('*.png')):
        print(f'    {f.name}')
    print(f'    amazon_reviews_clustered.csv')
    print('=' * 70)


if __name__ == '__main__':
    main()
