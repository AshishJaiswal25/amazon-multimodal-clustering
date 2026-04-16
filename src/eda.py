# src/eda.py
# Cells 7, 8, 9, 16 from your notebook
# Basic summary stats + all EDA figures + PCA scree plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from src.config import OUTPUT_DIR, RATING_COLS, CLUSTER_COLS, N_PCA, RANDOM_STATE
from src.preprocessing import STOP_WORDS_SET


# ── Cell 7: Key Statistics (text summary) ─────────────────────────────────
def print_summary_stats(df):
    print('=' * 55)
    print('  DATASET SUMMARY STATISTICS')
    print('=' * 55)
    print(f'  Total reviews (sample)   : {len(df):,}')
    print(f'  Average star rating      : {df["Score"].mean():.2f} / 5')
    print(f'  Median review length     : {df["word_count"].median():.0f} words')
    voted = (df['HelpfulnessDenominator'] > 0)
    print(f'  Reviews with any votes   : {voted.sum():,} ({voted.mean()*100:.1f}%)')
    print()
    print('  Rating breakdown:')
    for s in range(1, 6):
        n   = (df['Score'] == s).sum()
        pct = n / len(df) * 100
        bar = '█' * int(pct / 2)
        print(f'    {s}★  {bar:<25} {pct:5.1f}%  ({n:,})')


# ── Cell 8: EDA Figure 1 — Core Distributions ─────────────────────────────
def plot_eda_overview(df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Amazon Fine Food Reviews — Exploratory Data Analysis',
                 fontsize=15, fontweight='bold')

    # 1. Rating bar chart
    rc = df['Score'].value_counts().sort_index()
    axes[0, 0].bar(rc.index, rc.values, color=RATING_COLS, edgecolor='white', linewidth=1.5)
    for x, y in zip(rc.index, rc.values):
        axes[0, 0].text(x, y + max(rc.values)*0.01, f'{y/len(df)*100:.1f}%',
                        ha='center', fontsize=9)
    axes[0, 0].set_title('Rating Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Star Rating'); axes[0, 0].set_ylabel('Count')

    # 2. Review length histogram
    axes[0, 1].hist(df['word_count'].clip(0, 600), bins=70,
                    color='#5c6bc0', edgecolor='white', alpha=0.85)
    med = df['word_count'].median()
    axes[0, 1].axvline(med, color='red', linestyle='--', label=f'Median: {med:.0f} words')
    axes[0, 1].set_title('Review Word Count Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Word Count'); axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()

    # 3. Helpfulness ratio
    voted = df[df['HelpfulnessDenominator'] > 0]
    axes[0, 2].hist(voted['helpfulness_ratio'], bins=50,
                    color='#ab47bc', edgecolor='white', alpha=0.85)
    axes[0, 2].set_title(f'Helpfulness Ratio\n(n={len(voted):,} reviews with votes)',
                         fontweight='bold')
    axes[0, 2].set_xlabel('Helpful / Total Votes'); axes[0, 2].set_ylabel('Frequency')

    # 4. Reviews per year
    rpy = df.groupby('Year').size()
    axes[1, 0].bar(rpy.index, rpy.values, color='#ef5350', edgecolor='white', alpha=0.85)
    axes[1, 0].set_title('Reviews Over Time', fontweight='bold')
    axes[1, 0].set_xlabel('Year'); axes[1, 0].set_ylabel('Number of Reviews')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 5. Avg rating per year
    ary = df.groupby('Year')['Score'].mean()
    axes[1, 1].plot(ary.index, ary.values, 'o-', color='#ff7043', linewidth=2.5, markersize=8)
    axes[1, 1].set_ylim(1, 5)
    axes[1, 1].set_title('Avg Rating Over Time', fontweight='bold')
    axes[1, 1].set_xlabel('Year'); axes[1, 1].set_ylabel('Avg Star Rating')
    axes[1, 1].tick_params(axis='x', rotation=45)

    # 6. Box plot: word count by rating
    groups = [df[df['Score'] == s]['word_count'].clip(0, 500).values for s in range(1, 6)]
    bp = axes[1, 2].boxplot(groups, labels=[f'{s}★' for s in range(1, 6)],
                            patch_artist=True, notch=True, widths=0.5)
    for patch, c in zip(bp['boxes'], RATING_COLS):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    axes[1, 2].set_title('Review Length by Star Rating', fontweight='bold')
    axes[1, 2].set_xlabel('Star Rating'); axes[1, 2].set_ylabel('Word Count')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'eda_overview.png', dpi=130, bbox_inches='tight')
    plt.show()
    print('Saved: outputs/eda_overview.png')


# ── Cell 9: EDA Figure 2 — Text & Correlation Insights ───────────────────
def plot_eda_text_insights(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Text Content & Correlations', fontsize=14, fontweight='bold')

    # 1. Overall word cloud
    sample_size = min(5000, len(df))
    all_text = ' '.join(df['Text'].sample(sample_size, random_state=RANDOM_STATE))
    wc = WordCloud(width=700, height=450, background_color='white',
                   stopwords=STOP_WORDS_SET, colormap='plasma',
                   max_words=120, collocations=False).generate(all_text)
    axes[0].imshow(wc, interpolation='bilinear'); axes[0].axis('off')
    axes[0].set_title('Most Frequent Words (All Reviews)', fontweight='bold')

    # 2. % 5★ vs 1★ over time
    yearly = df.groupby('Year')['Score'].apply(
        lambda x: pd.Series({'pct_5star': (x == 5).mean() * 100,
                              'pct_1star': (x == 1).mean() * 100})
    ).unstack()
    yearly['pct_5star'].plot(ax=axes[1], color='#43a047', marker='o',
                              linewidth=2, label='5★')
    yearly['pct_1star'].plot(ax=axes[1], color='#e53935', marker='s',
                              linewidth=2, label='1★')
    axes[1].set_title('% of 5★ and 1★ Reviews Over Time', fontweight='bold')
    axes[1].set_xlabel('Year'); axes[1].set_ylabel('Percentage (%)')
    axes[1].legend(); axes[1].tick_params(axis='x', rotation=45)

    # 3. Helpfulness vs word count scatter
    voted = df[df['HelpfulnessDenominator'] > 2]
    s2    = voted.sample(min(3000, len(voted)), random_state=RANDOM_STATE)
    sc    = axes[2].scatter(s2['word_count'].clip(0, 700), s2['helpfulness_ratio'],
                            c=s2['Score'], cmap='RdYlGn', alpha=0.45, s=15, vmin=1, vmax=5)
    plt.colorbar(sc, ax=axes[2], label='Star Rating')
    axes[2].set_title('Helpfulness vs Review Length', fontweight='bold')
    axes[2].set_xlabel('Word Count'); axes[2].set_ylabel('Helpfulness Ratio')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'eda_text_insights.png', dpi=130, bbox_inches='tight')
    plt.show()
    print('Saved: outputs/eda_text_insights.png')


# ── Numerical correlation heatmap (from Cell 13) ──────────────────────────
def plot_numeric_correlation(X_num, use_cols):
    fig, ax = plt.subplots(figsize=(7, 5))
    cdf = pd.DataFrame(X_num, columns=use_cols)
    sns.heatmap(cdf.corr(), annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, square=True, linewidths=0.5, vmin=-1, vmax=1)
    ax.set_title('Numerical Features — Correlation Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'num_correlation.png', dpi=130, bbox_inches='tight')
    plt.show()
    print('Saved: outputs/num_correlation.png')


# ── Cell 16: PCA Scree + Cumulative Variance ──────────────────────────────
def plot_pca_analysis(pca):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('PCA — Dimensionality Reduction of Multimodal Features', fontweight='bold')

    evr = pca.explained_variance_ratio_
    axes[0].bar(range(1, N_PCA + 1), evr * 100, color='#5c6bc0', alpha=0.7, edgecolor='white')
    axes[0].plot(range(1, N_PCA + 1), evr * 100, 'o-', color='#283593', markersize=5)
    axes[0].set_title('Scree Plot — Variance per Component', fontweight='bold')
    axes[0].set_xlabel('Principal Component'); axes[0].set_ylabel('Explained Variance (%)')

    cumvar = np.cumsum(evr) * 100
    axes[1].plot(range(1, N_PCA + 1), cumvar, 'o-', color='#26a69a', linewidth=2.5, markersize=6)
    axes[1].fill_between(range(1, N_PCA + 1), cumvar, alpha=0.15, color='#26a69a')
    for thresh, col, lbl in [(80, 'red', '80%'), (90, 'orange', '90%')]:
        axes[1].axhline(thresh, color=col, linestyle='--', alpha=0.7, label=f'{lbl} threshold')
    axes[1].set_title('Cumulative Explained Variance', fontweight='bold')
    axes[1].set_xlabel('Number of PCs'); axes[1].set_ylabel('Cumulative Variance (%)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pca_analysis.png', dpi=130, bbox_inches='tight')
    plt.show()
    print('Saved: outputs/pca_analysis.png')
