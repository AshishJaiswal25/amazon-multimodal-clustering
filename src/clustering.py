# src/clustering.py
# Cells 18–27 from your notebook + Feature #3: k-distance DBSCAN tuning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from src.config import (
    OPTIMAL_K, RANDOM_STATE, OUTPUT_DIR,
    DBSCAN_MIN_SAMPLES, K_NEIGHBORS, CLUSTER_COLS
)


# ── Cell 18: Elbow + Silhouette ────────────────────────────────────────────
def find_optimal_k(X_pca, k_range=range(2, 11)):
    print('Searching for optimal k (K-Means)...')
    inertias, sil_scores = [], []

    for k in k_range:
        km  = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
        lbl = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_pca, lbl,
                               sample_size=min(8000, len(lbl)),
                               random_state=RANDOM_STATE)
        sil_scores.append(sil)
        print(f'  k={k}: inertia={km.inertia_:,.0f}  silhouette={sil:.4f}')

    best_k = list(k_range)[int(np.argmax(sil_scores))]
    print(f'\nBest k by silhouette: {best_k}')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('K-Means: Finding Optimal Number of Clusters', fontweight='bold')

    axes[0].plot(list(k_range), inertias, 'o-', color='#ef5350', linewidth=2.5, markersize=8)
    axes[0].set_title('Elbow Method (Inertia)', fontweight='bold')
    axes[0].set_xlabel('Number of Clusters k'); axes[0].set_ylabel('Inertia')
    axes[0].set_xticks(list(k_range))

    axes[1].plot(list(k_range), sil_scores, 'o-', color='#42a5f5', linewidth=2.5, markersize=8)
    axes[1].axvline(best_k, color='red', linestyle='--', label=f'Best k = {best_k}')
    axes[1].set_title('Silhouette Score vs k', fontweight='bold')
    axes[1].set_xlabel('Number of Clusters k'); axes[1].set_ylabel('Silhouette Score')
    axes[1].set_xticks(list(k_range)); axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'elbow_silhouette.png', dpi=130, bbox_inches='tight')
    plt.show()
    print('Saved: outputs/elbow_silhouette.png')
    return best_k, inertias, sil_scores


# ── Cell 19: Final K-Means ─────────────────────────────────────────────────
def run_kmeans(X_pca, k=OPTIMAL_K):
    print(f'Running final K-Means with k={k}...')
    km  = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20, max_iter=500)
    lbl = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, lbl,
                           sample_size=min(10000, len(lbl)),
                           random_state=RANDOM_STATE)
    print(f'  Final Silhouette Score: {sil:.4f}')
    print('\nCluster sizes:')
    sizes = pd.Series(lbl).value_counts().sort_index()
    for c, n in sizes.items():
        print(f'  Cluster {c}: {n:,} reviews ({n/len(lbl)*100:.1f}%)')
    return lbl, km, sil


# ── Feature #3: k-Distance Plot for DBSCAN eps tuning ─────────────────────
def kdistance_plot(X, min_samples=DBSCAN_MIN_SAMPLES):
    """
    Plots the sorted k-NN distance curve.
    The 'knee' (sharpest bend) gives the ideal eps for DBSCAN.
    Eliminates manual guessing.
    """
    print(f'\nBuilding k-distance plot (k={min_samples}) for DBSCAN eps selection...')
    nbrs = NearestNeighbors(n_neighbors=min_samples, n_jobs=-1)
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X)
    kth_dist = np.sort(distances[:, -1])[::-1]

    # Auto-detect knee via max second derivative in first half
    n  = len(kth_dist)
    x  = np.linspace(0, 1, n)
    y  = (kth_dist - kth_dist.min()) / (kth_dist.max() - kth_dist.min() + 1e-9)
    d2 = np.gradient(np.gradient(y, x), x)
    knee_idx     = int(np.argmax(d2[:n // 2]))
    suggested_eps = float(kth_dist[knee_idx])
    print(f'  Suggested eps (knee): {suggested_eps:.4f}')

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(kth_dist, color='#457b9d', linewidth=1.5, label=f'{min_samples}-NN distance')
    ax.axvline(knee_idx, color='red', linestyle='--', linewidth=1.5,
               label=f'Knee ≈ idx {knee_idx}')
    ax.axhline(suggested_eps, color='green', linestyle=':', linewidth=1.5,
               label=f'Suggested eps ≈ {suggested_eps:.3f}')
    ax.set_title(f'k-Distance Plot (k={min_samples}) — DBSCAN eps Selection',
                 fontweight='bold', fontsize=13)
    ax.set_xlabel('Points sorted by distance (desc)')
    ax.set_ylabel(f'Distance to {min_samples}-th nearest neighbour')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dbscan_kdistance.png', dpi=130, bbox_inches='tight')
    plt.show()
    print('Saved: outputs/dbscan_kdistance.png')
    return suggested_eps


# ── Cell 21: DBSCAN ────────────────────────────────────────────────────────
def run_dbscan(X_tsne, eps=None, min_samples=DBSCAN_MIN_SAMPLES):
    """
    Cell 21 from notebook runs DBSCAN on t-SNE coords.
    We add k-distance auto-tuning before that.
    """
    if eps is None:
        eps = kdistance_plot(X_tsne, min_samples=min_samples)

    print(f'\nRunning DBSCAN (eps={eps:.4f}, min_samples={min_samples})...')
    db  = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    lbl = db.fit_predict(X_tsne)

    n_clusters = len(set(lbl)) - (1 if -1 in lbl else 0)
    n_noise    = (lbl == -1).sum()
    print(f'  DBSCAN clusters found : {n_clusters}')
    print(f'  Noise / outlier points: {n_noise:,} ({n_noise/len(lbl)*100:.1f}%)')
    return lbl, db, n_clusters, n_noise


# ── Cell 22: Agglomerative ─────────────────────────────────────────────────
def run_agglomerative(X_fused, k=OPTIMAL_K):
    print('Running Agglomerative Clustering (Ward linkage)...')
    pca10   = PCA(n_components=min(10, X_fused.shape[1]), random_state=RANDOM_STATE)
    X_pca10 = pca10.fit_transform(X_fused)

    agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
    lbl = agg.fit_predict(X_pca10)
    sil = silhouette_score(X_pca10, lbl,
                           sample_size=min(10000, len(lbl)),
                           random_state=RANDOM_STATE)
    print(f'  Agglomerative silhouette: {sil:.4f}')
    return lbl, agg, sil, X_pca10


# ── Cells 24–25: Louvain Graph Clustering ─────────────────────────────────
def run_louvain(X_pca):
    try:
        import community as community_louvain
        import networkx as nx
    except ImportError:
        print('Install: pip install python-louvain networkx')
        return np.zeros(len(X_pca), dtype=int), None, 1, 0.0, 0.0

    print(f'\nStep 1 — Building k-NN graph (k={K_NEIGHBORS}) on {len(X_pca):,} nodes...')
    X_norm = normalize(X_pca, norm='l2')
    A = kneighbors_graph(X_norm, n_neighbors=K_NEIGHBORS, mode='distance',
                         metric='euclidean', include_self=False, n_jobs=-1)
    A_sim      = A.copy()
    A_sim.data = np.clip(1.0 - (A.data ** 2) / 2, 0, 1)
    A_sym      = A_sim.maximum(A_sim.T)
    print(f'  {A_sym.shape[0]:,} nodes | {A_sym.nnz // 2:,} edges')

    print('Step 2 — Converting to NetworkX & running Louvain...')
    try:
        G = nx.from_scipy_sparse_array(A_sym, edge_attribute='weight')
    except AttributeError:
        G = nx.from_scipy_sparse_matrix(A_sym, edge_attribute='weight')

    partition    = community_louvain.best_partition(G, weight='weight',
                                                    random_state=RANDOM_STATE)
    labels       = np.array([partition[i] for i in range(len(X_pca))])
    n_communities = len(set(labels))
    modularity   = community_louvain.modularity(partition, G, weight='weight')
    graph_sil    = silhouette_score(X_pca, labels,
                                    sample_size=min(10000, len(labels)),
                                    random_state=RANDOM_STATE)

    print(f'\nLouvain Results')
    print(f'  Communities found: {n_communities}')
    print(f'  Modularity Q     : {modularity:.4f}  [>0.3 = meaningful structure]')
    print(f'  Silhouette score : {graph_sil:.4f}')
    sizes = pd.Series(labels).value_counts().sort_index()
    for c, n in sizes.items():
        print(f'  Community {c:>2}: {n:>6,} reviews ({n/len(labels)*100:.1f}%)')

    return labels, partition, n_communities, modularity, graph_sil


# ── Cell 26: Network Graph Visualization ───────────────────────────────────
def plot_network_graph(df, n_comm, n_clusters=OPTIMAL_K, final_sil=0.0):
    import matplotlib.pyplot as plt
    cmap     = plt.cm.tab10 if n_comm <= 10 else plt.cm.tab20
    g_colors = [cmap(i / max(n_comm - 1, 1)) for i in range(n_comm)]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle('Network Graph Clustering — Louvain Community Detection',
                 fontweight='bold', fontsize=14)

    # Panel 1 — Louvain communities
    for c in range(n_comm):
        mask = df['graph_cluster'] == c
        axes[0].scatter(df.loc[mask, 'tsne_x'], df.loc[mask, 'tsne_y'],
                        s=0.5, alpha=0.4, color=g_colors[c], label=f'G{c}')
    axes[0].set_title(f'Louvain Communities (n={n_comm})', fontweight='bold')
    axes[0].set_xlabel('t-SNE 1'); axes[0].set_ylabel('t-SNE 2')
    if n_comm <= 12:
        axes[0].legend(markerscale=6, loc='upper right', fontsize=8, ncol=2)

    # Panel 2 — K-Means for comparison
    for c in range(n_clusters):
        mask = df['cluster'] == c
        axes[1].scatter(df.loc[mask, 'tsne_x'], df.loc[mask, 'tsne_y'],
                        s=0.5, alpha=0.4, color=CLUSTER_COLS[c % len(CLUSTER_COLS)],
                        label=f'K{c}')
    axes[1].set_title(f'K-Means (k={n_clusters}) — for comparison', fontweight='bold')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].legend(markerscale=6, loc='upper right', fontsize=8)

    # Panel 3 — Community size bars
    comm_sizes = df['graph_cluster'].value_counts().sort_index()
    bars = axes[2].bar(range(n_comm), comm_sizes.values,
                       color=g_colors, edgecolor='white', linewidth=0.5)
    axes[2].set_title('Community Size Distribution', fontweight='bold')
    axes[2].set_xlabel('Community ID'); axes[2].set_ylabel('Number of Reviews')
    axes[2].set_xticks(range(n_comm))
    axes[2].set_xticklabels([f'G{i}' for i in comm_sizes.index], rotation=45, ha='right')
    for bar, val in zip(bars, comm_sizes.values):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(comm_sizes)*0.01,
                     f'{val/len(df)*100:.1f}%', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'network_graph_clustering.png', dpi=130, bbox_inches='tight')
    plt.show()
    print('Saved: outputs/network_graph_clustering.png')


# ── Cell 27: Algorithm Comparison Table ────────────────────────────────────
def print_algorithm_comparison(df, final_sil, agg_sil, graph_sil,
                                n_communities, modularity, n_clusters=OPTIMAL_K):
    graph_profile = df.groupby('graph_cluster').agg(
        n_reviews       = ('graph_cluster', 'count'),
        avg_rating      = ('Score', 'mean'),
        pct_5star       = ('Score', lambda x: (x == 5).mean() * 100),
        pct_1star       = ('Score', lambda x: (x == 1).mean() * 100),
        avg_words       = ('word_count', 'mean'),
        avg_helpfulness = ('helpfulness_ratio', 'mean'),
    ).round(2)
    graph_profile['pct_reviews'] = (graph_profile['n_reviews'] / len(df) * 100).round(1)
    print('GRAPH COMMUNITY PROFILES')
    print('=' * 70)
    print(graph_profile.to_string())

    ari_graph_km  = adjusted_rand_score(df['cluster'],     df['graph_cluster'])
    ari_graph_agg = adjusted_rand_score(df['agg_cluster'], df['graph_cluster'])
    ari_km_agg    = adjusted_rand_score(df['cluster'],     df['agg_cluster'])

    print('\n' + '=' * 55)
    print('  CLUSTERING ALGORITHM COMPARISON')
    print('=' * 55)
    print(f'  {"Method":<30} {"Silhouette":>10} {"# Clusters":>11}')
    print('-' * 55)
    print(f'  {"K-Means (k="+str(n_clusters)+")":<30} {final_sil:>10.4f} {n_clusters:>11}')
    print(f'  {"Agglomerative (Ward)":<30} {agg_sil:>10.4f} {n_clusters:>11}')
    print(f'  {"Louvain (Graph)":<30} {graph_sil:>10.4f} {n_communities:>11}')
    print('=' * 55)
    print(f'\n  ARI  K-Means  vs Agglomerative : {ari_km_agg:.4f}')
    print(f'  ARI  K-Means  vs Graph         : {ari_graph_km:.4f}')
    print(f'  ARI  Agg      vs Graph         : {ari_graph_agg:.4f}')
    print(f'  Modularity Q (Graph)           : {modularity:.4f}')
    print('\n  [ARI: 1=identical, 0=random | Q>0.3 = meaningful structure]')
