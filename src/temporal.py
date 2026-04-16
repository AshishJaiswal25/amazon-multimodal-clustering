# src/temporal.py — Feature #2: Temporal Clustering Analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from src.config import OUTPUT_DIR, CLUSTER_COLS


def plot_temporal_drift(df, cluster_col='cluster'):
    year_counts = df['Year'].value_counts()
    valid_years = year_counts[year_counts >= 30].index
    df_v = df[df['Year'].isin(valid_years)].copy()

    yearly = df_v.groupby(['Year', cluster_col]).size().unstack(fill_value=0)
    yearly_pct = yearly.div(yearly.sum(axis=1), axis=0) * 100
    n = df[cluster_col].nunique()
    colors = CLUSTER_COLS[:n]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle('Temporal Cluster Dynamics — How Review Patterns Shift Over Time',
                 fontweight='bold', fontsize=14)

    # Stacked area
    bottom = np.zeros(len(yearly_pct))
    for i, c in enumerate(yearly_pct.columns):
        vals = yearly_pct[c].values
        axes[0].fill_between(yearly_pct.index, bottom, bottom + vals,
                             alpha=0.75, color=colors[i % len(colors)], label=f'C{c}')
        bottom += vals
    axes[0].set_title('Cluster Composition Over Time', fontweight='bold')
    axes[0].set_xlabel('Year'); axes[0].set_ylabel('% of Reviews')
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[0].legend(loc='upper left', fontsize=8)
    axes[0].grid(alpha=0.2)

    # Avg rating per cluster over time
    rt = df_v.groupby(['Year', cluster_col])['Score'].mean().unstack()
    for i, c in enumerate(rt.columns):
        axes[1].plot(rt.index, rt[c], 'o-', color=colors[i % len(colors)],
                     linewidth=2, markersize=6, label=f'C{c}')
    axes[1].set_title('Avg Rating per Cluster Over Time', fontweight='bold')
    axes[1].set_xlabel('Year'); axes[1].set_ylabel('Average Star Rating')
    axes[1].set_ylim(1, 5.2); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.2)

    # Heatmap
    sns.heatmap(yearly_pct.T, annot=True, fmt='.1f', cmap='YlOrRd',
                linewidths=0.5, ax=axes[2], cbar_kws={'label': '% of reviews'})
    axes[2].set_title('Cluster Share Heatmap by Year', fontweight='bold')
    axes[2].set_xlabel('Year'); axes[2].set_ylabel('Cluster')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'temporal_clustering.png', dpi=130, bbox_inches='tight')
    plt.show()
    print('Saved: outputs/temporal_clustering.png')


def seasonal_complaint_analysis(df, cluster_col='cluster'):
    cluster_ratings = df.groupby(cluster_col)['Score'].mean()
    complaint_c     = cluster_ratings.idxmin()
    print(f'\nComplaint cluster (lowest avg rating): Cluster {complaint_c} '
          f'({cluster_ratings[complaint_c]:.2f}★)')

    monthly = (df[df[cluster_col] == complaint_c]
               .groupby('Month').size()
               .reindex(range(1, 13), fill_value=0))
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(month_names, monthly.values,
                  color=['#e63946' if m in [11, 12, 1] else '#457b9d'
                         for m in range(1, 13)],
                  edgecolor='white', linewidth=1.2)
    ax.set_title(f'Complaint Cluster {complaint_c} — Monthly Review Volume\n'
                 '(Red = Nov/Dec/Jan holiday window)', fontweight='bold', fontsize=13)
    ax.set_xlabel('Month'); ax.set_ylabel('Number of Reviews')
    for bar, val in zip(bars, monthly.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(monthly)*0.01,
                str(val), ha='center', va='bottom', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'seasonal_complaints.png', dpi=130, bbox_inches='tight')
    plt.show()
    print('Saved: outputs/seasonal_complaints.png')
