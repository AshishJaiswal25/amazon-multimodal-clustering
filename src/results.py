# src/results.py
# Cells 29–35 from your notebook
# t-SNE viz, cluster profiles, word clouds, top terms, algo comparison,
# sample reviews, final dashboard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import OUTPUT_DIR, CLUSTER_COLS, RATING_COLS, OPTIMAL_K, N_PCA, RANDOM_STATE
from src.preprocessing import STOP_WORDS_SET


# ── Cell 29: t-SNE — K-Means vs Rating vs Review Length ───────────────────
def plot_tsne_results(df, optimal_k=OPTIMAL_K):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f't-SNE Map of {len(df):,} Reviews', fontsize=14, fontweight='bold')

    for c in range(optimal_k):
        m = df['cluster'] == c
        axes[0].scatter(df.loc[m, 'tsne_x'], df.loc[m, 'tsne_y'],
                        s=3, alpha=0.4, color=CLUSTER_COLS[c % len(CLUSTER_COLS)],
                        label=f'Cluster {c}')
    axes[0].set_title(f'K-Means (k={optimal_k})', fontweight='bold')
    axes[0].legend(markerscale=5, fontsize=9)
    axes[0].set_xlabel('t-SNE 1'); axes[0].set_ylabel('t-SNE 2')

    sc = axes[1].scatter(df['tsne_x'], df['tsne_y'], c=df['Score'],
                         cmap='RdYlGn', s=3, alpha=0.35, vmin=1, vmax=5)
    plt.colorbar(sc, ax=axes[1], label='Star Rating')
    axes[1].set_title('Colored by Star Rating', fontweight='bold')
    axes[1].set_xlabel('t-SNE 1'); axes[1].set_ylabel('t-SNE 2')

    sc2 = axes[2].scatter(df['tsne_x'], df['tsne_y'], c=df['log_word_count'],
                          cmap='YlOrBr', s=3, alpha=0.35)
    plt.colorbar(sc2, ax=axes[2], label='log(Word Count)')
    axes[2].set_title('Colored by Review Length', fontweight='bold')
    axes[2].set_xlabel('t-SNE 1'); axes[2].set_ylabel('t-SNE 2')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'tsne_visualization.png', dpi=130, bbox_inches='tight')
    plt.show()
    print('Saved: outputs/tsne_visualization.png')


# ── Cell 30: Cluster Profile Heatmap + Stacked Rating Bars ────────────────
def plot_cluster_profiles(df, optimal_k=OPTIMAL_K):
    profile = df.groupby('cluster').agg(
        n_reviews       = ('Id', 'count'),
        avg_rating      = ('Score', 'mean'),
        pct_5star       = ('Score', lambda x: (x == 5).mean() * 100),
        pct_1star       = ('Score', lambda x: (x == 1).mean() * 100),
        avg_words       = ('word_count', 'mean'),
        avg_helpfulness = ('helpfulness_ratio', 'mean'),
        avg_votes       = ('HelpfulnessDenominator', 'mean'),
    ).round(2)

    print('Cluster Profiles:')
    print(profile.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Cluster Profiles', fontweight='bold')

    cols_for_heat = ['avg_rating', 'pct_5star', 'pct_1star',
                     'avg_words', 'avg_helpfulness', 'avg_votes']
    heat_data = profile[cols_for_heat].copy()
    heat_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min())
    sns.heatmap(heat_norm.T, annot=heat_data.T.round(1), fmt='g',
                cmap='YlOrRd', ax=axes[0], linewidths=0.5,
                cbar_kws={'label': 'Normalized (0-1)'})
    axes[0].set_title('Normalized Feature Heatmap per Cluster', fontweight='bold')
    axes[0].set_xlabel('Cluster')

    rating_dist = df.groupby(['cluster', 'Score']).size().unstack(fill_value=0)
    rating_pct  = rating_dist.div(rating_dist.sum(axis=1), axis=0) * 100
    rating_pct.plot(kind='bar', ax=axes[1], color=RATING_COLS,
                    edgecolor='white', width=0.7, stacked=True)
    axes[1].set_title('Rating Distribution per Cluster (%)', fontweight='bold')
    axes[1].set_xlabel('Cluster'); axes[1].set_ylabel('Percentage (%)')
    axes[1].legend(title='Stars', labels=['1★', '2★', '3★', '4★', '5★'],
                   bbox_to_anchor=(1.01, 1))
    axes[1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cluster_profiles.png', dpi=130, bbox_inches='tight')
    plt.show()
    print('Saved: outputs/cluster_profiles.png')
    return profile, rating_pct


# ── Cell 31: Word Clouds per Cluster ──────────────────────────────────────
def plot_wordclouds(df, optimal_k=OPTIMAL_K):
    cmaps = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples',
             'cool', 'hot', 'winter', 'autumn']

    fig, axes = plt.subplots(1, optimal_k, figsize=(5 * optimal_k, 5))
    if optimal_k == 1:
        axes = [axes]
    fig.suptitle('Word Clouds — Most Characteristic Terms per Cluster',
                 fontsize=14, fontweight='bold')

    for c in range(optimal_k):
        cdf   = df[df['cluster'] == c]
        texts = ' '.join(cdf['clean_text'].sample(min(3000, len(cdf)), random_state=RANDOM_STATE))
        wc = WordCloud(width=420, height=300, background_color='white',
                       colormap=cmaps[c % len(cmaps)], max_words=80,
                       stopwords=STOP_WORDS_SET, collocations=False).generate(texts)
        axes[c].imshow(wc, interpolation='bilinear'); axes[c].axis('off')
        axes[c].set_title(f'Cluster {c}\nn={len(cdf):,}  avg={cdf["Score"].mean():.2f}★',
                          fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cluster_wordclouds.png', dpi=130, bbox_inches='tight')
    plt.show()
    print('Saved: outputs/cluster_wordclouds.png')


# ── Cell 32: Top TF-IDF Terms per Cluster ─────────────────────────────────
def print_top_terms(df, optimal_k=OPTIMAL_K):
    print('TOP DISCRIMINATIVE TERMS PER CLUSTER')
    print('=' * 70)
    for c in range(optimal_k):
        cdf      = df[df['cluster'] == c]
        tfidf_c  = TfidfVectorizer(max_features=3000, stop_words='english',
                                   ngram_range=(1, 2))
        tfidf_c.fit(cdf['clean_text'].tolist())
        feat_names = tfidf_c.get_feature_names_out()
        top_idx    = np.argsort(tfidf_c.idf_)[:15]
        top_terms  = [feat_names[i] for i in top_idx]
        print(f'\nCluster {c}  (n={len(cdf):,}, avg_rating={cdf["Score"].mean():.2f}★, '
              f'avg_words={cdf["word_count"].mean():.0f})')
        print('  Top terms:')
        for t in top_terms:
            print(f'    {t}')


# ── Cell 33: Algorithm Comparison — 3-panel t-SNE ─────────────────────────
def plot_algorithm_comparison(df, final_sil, agg_sil, n_db_clusters, n_noise,
                               optimal_k=OPTIMAL_K):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Clustering Algorithm Comparison on t-SNE Map',
                 fontsize=14, fontweight='bold')

    # K-Means
    for c in range(optimal_k):
        m = df['cluster'] == c
        axes[0].scatter(df.loc[m, 'tsne_x'], df.loc[m, 'tsne_y'],
                        s=3, alpha=0.4, color=CLUSTER_COLS[c % len(CLUSTER_COLS)],
                        label=f'C{c}')
    axes[0].set_title(f'K-Means (k={optimal_k})\nSilhouette: {final_sil:.4f}',
                      fontweight='bold')
    axes[0].legend(markerscale=5, fontsize=9)

    # DBSCAN
    db_ids  = sorted(set(df['dbscan_cluster']))
    cmap_db = plt.cm.get_cmap('tab10', len(db_ids))
    for i, cid in enumerate(db_ids):
        m     = df['dbscan_cluster'] == cid
        color = 'lightgrey' if cid == -1 else cmap_db(i)
        lbl   = 'Noise' if cid == -1 else f'C{cid}'
        axes[1].scatter(df.loc[m, 'tsne_x'], df.loc[m, 'tsne_y'],
                        s=3, alpha=0.35, color=color, label=lbl)
    axes[1].set_title(
        f'DBSCAN\n{n_db_clusters} clusters, {n_noise/len(df)*100:.1f}% noise',
        fontweight='bold')
    axes[1].legend(markerscale=5, fontsize=8, ncol=2)

    # Agglomerative
    for c in range(optimal_k):
        m = df['agg_cluster'] == c
        axes[2].scatter(df.loc[m, 'tsne_x'], df.loc[m, 'tsne_y'],
                        s=3, alpha=0.4, color=CLUSTER_COLS[c % len(CLUSTER_COLS)],
                        label=f'C{c}')
    ari = 0.0
    try:
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(df['cluster'], df['agg_cluster'])
    except Exception:
        pass
    axes[2].set_title(f'Agglomerative Ward\nSilhouette: {agg_sil:.4f}  ARI: {ari:.4f}',
                      fontweight='bold')
    axes[2].legend(markerscale=5, fontsize=9)

    for ax in axes:
        ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'algorithm_comparison.png', dpi=130, bbox_inches='tight')
    plt.show()
    print('Saved: outputs/algorithm_comparison.png')


# ── Cell 34: Sample Reviews per Cluster ───────────────────────────────────
def print_sample_reviews(df, optimal_k=OPTIMAL_K):
    print('REPRESENTATIVE REVIEWS PER CLUSTER')
    print('=' * 80)
    for c in range(optimal_k):
        cdf = df[df['cluster'] == c]
        print(f'\n{"─"*80}')
        print(f'CLUSTER {c}  |  {len(cdf):,} reviews  |  avg={cdf["Score"].mean():.2f}★  '
              f'|  avg_words={cdf["word_count"].mean():.0f}')
        print(f'{"─"*80}')
        for _, row in cdf.sample(min(2, len(cdf)), random_state=RANDOM_STATE).iterrows():
            print(f'  [{row["Score"]}★]  {row["Summary"]}')
            print(f'  "{str(row["Text"])[:220]}..."')
            print()


# ── Cell 35: Final Summary Dashboard ──────────────────────────────────────
def plot_final_dashboard(df, profile, rating_pct, pca, final_sil, agg_sil,
                          optimal_k=OPTIMAL_K):
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Multimodal Clustering — Amazon Fine Food Reviews\nFinal Results Dashboard',
                 fontsize=17, fontweight='bold', y=1.01)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    # 1. t-SNE cluster map (large)
    ax1 = fig.add_subplot(gs[0, :2])
    for c in range(optimal_k):
        m = df['cluster'] == c
        ax1.scatter(df.loc[m, 'tsne_x'], df.loc[m, 'tsne_y'],
                    s=5, alpha=0.45, color=CLUSTER_COLS[c % len(CLUSTER_COLS)],
                    label=f'Cluster {c}')
    ax1.set_title(f't-SNE Multimodal Cluster Map  (Silhouette={final_sil:.4f})',
                  fontweight='bold')
    ax1.legend(markerscale=4, fontsize=10)
    ax1.set_xlabel('t-SNE 1'); ax1.set_ylabel('t-SNE 2')

    # 2. Cluster sizes with avg rating
    ax2 = fig.add_subplot(gs[0, 2])
    bars = ax2.bar(range(optimal_k), profile['n_reviews'],
                   color=CLUSTER_COLS[:optimal_k], edgecolor='white', alpha=0.9)
    ax2r = ax2.twinx()
    ax2r.plot(range(optimal_k), profile['avg_rating'], 'D--',
              color='black', linewidth=2, markersize=9, label='Avg ★')
    ax2.set_title('Cluster Size & Avg Rating', fontweight='bold')
    ax2.set_xlabel('Cluster'); ax2.set_ylabel('# Reviews')
    ax2r.set_ylabel('Avg Rating'); ax2r.set_ylim(1, 5)
    ax2r.legend(loc='upper right')
    for bar, n in zip(bars, profile['n_reviews']):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(profile['n_reviews']) * 0.01,
                 f'{n:,}', ha='center', va='bottom', fontsize=9)

    # 3. Algorithm silhouette comparison
    ax3 = fig.add_subplot(gs[1, 0])
    methods = ['K-Means', 'Agglomerative']
    scores  = [final_sil, agg_sil]
    colors3 = ['#5c6bc0', '#26a69a']
    ax3.barh(methods, scores, color=colors3, edgecolor='white', height=0.5)
    for i, s in enumerate(scores):
        ax3.text(s + 0.001, i, f'{s:.4f}', va='center', fontsize=10)
    ax3.set_title('Silhouette Score Comparison', fontweight='bold')
    ax3.set_xlabel('Silhouette Score')
    ax3.set_xlim(0, max(scores) * 1.25)

    # 4. PCA cumulative variance
    ax4 = fig.add_subplot(gs[1, 1])
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    ax4.plot(range(1, N_PCA + 1), cumvar, 'o-', color='#ef5350', linewidth=2.5, markersize=5)
    ax4.fill_between(range(1, N_PCA + 1), cumvar, alpha=0.15, color='#ef5350')
    ax4.axhline(80, color='grey', linestyle='--', alpha=0.6)
    ax4.set_title('PCA Cumulative Variance', fontweight='bold')
    ax4.set_xlabel('# Principal Components'); ax4.set_ylabel('Var Explained (%)')

    # 5. Stacked rating bars per cluster
    ax5 = fig.add_subplot(gs[1, 2])
    rating_pct.plot(kind='bar', ax=ax5, color=RATING_COLS,
                    edgecolor='white', stacked=True, width=0.7)
    ax5.set_title('Star Rating Mix per Cluster', fontweight='bold')
    ax5.set_xlabel('Cluster'); ax5.set_ylabel('%')
    ax5.legend(labels=['1★', '2★', '3★', '4★', '5★'],
               fontsize=8, bbox_to_anchor=(1.01, 1))
    ax5.tick_params(axis='x', rotation=0)

    plt.savefig(OUTPUT_DIR / 'final_dashboard.png', dpi=130, bbox_inches='tight')
    plt.show()
    print('Dashboard saved: outputs/final_dashboard.png')
