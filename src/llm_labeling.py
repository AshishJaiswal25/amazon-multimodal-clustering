# src/llm_labeling.py — Feature #5: LLM Auto-Labeling via Claude API
import json, time, urllib.request, urllib.error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import OUTPUT_DIR, LLM_MODEL, LLM_MAX_TOKENS, LLM_TOP_TERMS, LLM_N_REVIEWS, CLUSTER_COLS


def get_top_terms(df, cluster_col='cluster', text_col='clean_text', n=LLM_TOP_TERMS):
    top_terms = {}
    for c in sorted(df[cluster_col].unique()):
        texts = df[df[cluster_col] == c][text_col].tolist()
        if len(texts) < 5:
            top_terms[c] = ['insufficient data']; continue
        try:
            tv = TfidfVectorizer(max_features=2000, stop_words='english',
                                 ngram_range=(1, 2), min_df=2)
            tv.fit(texts)
            feat = tv.get_feature_names_out()
            top_terms[c] = [feat[i] for i in np.argsort(tv.idf_)[:n]]
        except Exception:
            top_terms[c] = ['error']
    return top_terms


def build_prompt(c, top_terms, sample_reviews, avg_rating, n_reviews):
    revs = '\n'.join([f'  - [{r["score"]}★] "{r["text"][:200]}..."' for r in sample_reviews])
    return f"""You are analyzing a cluster of Amazon product reviews. Generate:
1. A short label (3-6 words) capturing the cluster essence
2. A one-sentence description (max 25 words)
3. Dominant sentiment: Positive / Negative / Mixed / Neutral

Cluster {c} info:
- Reviews: {n_reviews:,}  |  Avg rating: {avg_rating:.2f}/5.0
- Top terms: {', '.join(top_terms[:12])}
- Sample reviews:
{revs}

Respond ONLY with valid JSON, no other text:
{{"label": "...", "description": "...", "sentiment": "..."}}"""


def call_claude(prompt):
    payload = json.dumps({
        'model': LLM_MODEL, 'max_tokens': LLM_MAX_TOKENS,
        'messages': [{'role': 'user', 'content': prompt}]
    }).encode()
    req = urllib.request.Request(
        'https://api.anthropic.com/v1/messages', data=payload,
        headers={'Content-Type': 'application/json', 'anthropic-version': '2023-06-01'},
        method='POST')
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data    = json.loads(r.read().decode())
            content = data['content'][0]['text'].strip()
            content = content.replace('```json','').replace('```','').strip()
            return json.loads(content)
    except Exception as e:
        return {'label': 'Unknown', 'description': str(e)[:60], 'sentiment': 'Unknown'}


def auto_label_clusters(df, cluster_col='cluster', text_col='clean_text'):
    print('\n── LLM Auto-Labeling via Claude API ────────────────────────────')
    top_terms = get_top_terms(df, cluster_col, text_col)
    labels    = {}

    for c in sorted(df[cluster_col].unique()):
        cdf = df[df[cluster_col] == c]
        print(f'\n  Labeling Cluster {c} ({len(cdf):,} reviews)...')
        sample = cdf.sample(min(LLM_N_REVIEWS, len(cdf)), random_state=42)
        sample_reviews = [{'score': row['Score'], 'text': str(row['Text'])[:300]}
                          for _, row in sample.iterrows()]
        prompt = build_prompt(c, top_terms.get(c, []), sample_reviews,
                              cdf['Score'].mean(), len(cdf))
        result = call_claude(prompt)
        labels[c] = result
        print(f'  → Label    : {result.get("label")}')
        print(f'  → Desc     : {result.get("description")}')
        print(f'  → Sentiment: {result.get("sentiment")}')
        time.sleep(0.5)

    df['cluster_label']       = df[cluster_col].map(lambda c: labels.get(c, {}).get('label', f'C{c}'))
    df['cluster_description'] = df[cluster_col].map(lambda c: labels.get(c, {}).get('description', ''))
    df['cluster_sentiment']   = df[cluster_col].map(lambda c: labels.get(c, {}).get('sentiment', ''))
    print('\n✅ LLM labeling complete.')
    return labels


def plot_labeled_clusters(df, labels, cluster_col='cluster'):
    clusters = sorted(df[cluster_col].unique())
    n = len(clusters)
    colors = CLUSTER_COLS[:n]

    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1: axes = [axes]
    fig.suptitle('Cluster Profiles — Auto-Labeled by LLM', fontweight='bold', fontsize=14)

    for i, c in enumerate(clusters):
        ax  = axes[i]; cdf = df[df[cluster_col] == c]
        info = labels.get(c, {})
        ax.set_facecolor(colors[i] + '22')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        circle = plt.Circle((0.5, 0.82), 0.10, color=colors[i], zorder=3)
        ax.add_patch(circle)
        ax.text(0.5, 0.82, str(c), ha='center', va='center',
                fontsize=16, fontweight='bold', color='white', zorder=4)
        ax.text(0.5, 0.67, info.get('label', f'Cluster {c}'), ha='center',
                fontsize=11, fontweight='bold', color='#1a1a2e')
        ax.text(0.5, 0.54, f'⭐ {cdf["Score"].mean():.2f} / 5.0', ha='center', fontsize=10)
        ax.text(0.5, 0.44, f'📝 {len(cdf):,} reviews', ha='center', fontsize=10)
        ax.text(0.5, 0.34, f'😊 {info.get("sentiment","")}', ha='center',
                fontsize=10, style='italic', color='#555')
        desc  = info.get('description', '')
        words = desc.split(); lines = []; line = []
        for w in words:
            if sum(len(x) for x in line) + len(line) + len(w) > 28:
                lines.append(' '.join(line)); line = [w]
            else:
                line.append(w)
        if line: lines.append(' '.join(line))
        for j, ln in enumerate(lines[:3]):
            ax.text(0.5, 0.20 - j*0.09, ln, ha='center', fontsize=8, color='#555')
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color(colors[i]); spine.set_linewidth(2)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'llm_cluster_labels.png', dpi=130, bbox_inches='tight')
    plt.show()
    print('Saved: outputs/llm_cluster_labels.png')
