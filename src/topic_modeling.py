# src/topic_modeling.py — Feature #4: LDA Topic Modeling per Cluster
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from src.config import OUTPUT_DIR, LDA_N_TOPICS, LDA_MAX_ITER, RANDOM_STATE


def run_lda_per_cluster(df, cluster_col='cluster', text_col='clean_text',
                         n_topics=LDA_N_TOPICS, n_top_words=10):
    print('\n── Topic Modeling per Cluster (LDA) ──────────────────────────')
    clusters   = sorted(df[cluster_col].unique())
    all_topics = {}

    for c in clusters:
        cdf   = df[df[cluster_col] == c]
        texts = cdf[text_col].tolist()
        print(f'\nCluster {c}  ({len(texts):,} reviews)')
        if len(texts) < 20:
            print('  ⚠ Too few reviews, skipping.')
            continue
        try:
            cv = CountVectorizer(max_features=2000, min_df=2, max_df=0.90,
                                 stop_words='english', ngram_range=(1, 2))
            X_cv = cv.fit_transform(texts)
        except ValueError:
            print('  ⚠ Vocabulary too small, skipping.')
            continue

        actual_topics = min(n_topics, max(2, len(texts) // 50))
        lda = LatentDirichletAllocation(
            n_components=actual_topics, max_iter=LDA_MAX_ITER,
            learning_method='online', random_state=RANDOM_STATE, n_jobs=-1)
        lda.fit(X_cv)

        feat_names     = cv.get_feature_names_out()
        cluster_topics = []
        for t_idx, topic in enumerate(lda.components_):
            top_words = [feat_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            cluster_topics.append((t_idx, top_words))
            print(f'  Sub-topic {t_idx}: {", ".join(top_words)}')

        all_topics[c] = cluster_topics
        topic_dist = lda.transform(X_cv)
        df.loc[df[cluster_col] == c, 'sub_topic'] = topic_dist.argmax(axis=1)

    print('\n✅ LDA complete.')
    return all_topics


def plot_topic_heatmap(all_topics):
    if not all_topics:
        return
    rows = []
    for c, topics in all_topics.items():
        for t_idx, words in topics:
            rows.append({'cluster': f'C{c}', 'sub_topic': f'T{t_idx}',
                         'top_words': ' | '.join(words[:6])})
    df_t  = pd.DataFrame(rows)
    pivot = df_t.pivot(index='cluster', columns='sub_topic', values='top_words').fillna('')

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns)*3), max(4, len(pivot)*1.2)))
    ax.axis('off')
    tbl = ax.table(cellText=pivot.values, rowLabels=pivot.index,
                   colLabels=pivot.columns, cellLoc='left', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 2.2)
    ax.set_title('LDA Sub-Topics per Cluster (Top 6 Words)',
                 fontweight='bold', fontsize=13, pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'topic_modeling_per_cluster.png', dpi=130, bbox_inches='tight')
    plt.show()
    print('Saved: outputs/topic_modeling_per_cluster.png')


def print_topic_summary(all_topics):
    print('\n' + '=' * 70)
    print('  TOPIC MODELING SUMMARY')
    print('=' * 70)
    for c, topics in all_topics.items():
        print(f'\n  Cluster {c}:')
        for t_idx, words in topics:
            print(f'    Sub-topic {t_idx}: {", ".join(words)}')
