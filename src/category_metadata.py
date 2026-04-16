# src/category_metadata.py — Feature #1: Product Category Metadata
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import OUTPUT_DIR


def infer_category(row):
    text = (str(row.get('Summary','')) + ' ' + str(row.get('Text',''))).lower()
    if any(w in text for w in ['coffee','espresso','latte','brew','tea','chai']):
        return 'Coffee & Tea'
    if any(w in text for w in ['chocolate','candy','gummy','caramel']):
        return 'Candy & Chocolate'
    if any(w in text for w in ['nut','almond','cashew','peanut','chip','cracker','popcorn']):
        return 'Nuts & Snacks'
    if any(w in text for w in ['sauce','ketchup','mustard','salsa','dressing','vinegar','oil','spice']):
        return 'Condiments & Sauces'
    if any(w in text for w in ['pasta','rice','grain','quinoa','oat','cereal','bread','flour']):
        return 'Grains & Pasta'
    if any(w in text for w in ['vitamin','protein','supplement','probiotic','omega']):
        return 'Health & Supplements'
    if any(w in text for w in ['dog','cat','pet','puppy','kitten','kibble']):
        return 'Pet Food'
    if any(w in text for w in ['juice','drink','water','soda','beverage','smoothie']):
        return 'Beverages'
    if any(w in text for w in ['milk','cream','cheese','butter','yogurt','dairy']):
        return 'Dairy & Alternatives'
    if any(w in text for w in ['sugar','honey','syrup','jam','jelly','cookie','cake','baking']):
        return 'Baking & Sweeteners'
    if any(w in text for w in ['soup','broth','stock','chili','stew']):
        return 'Soups & Broths'
    return 'General Grocery'


def enrich_with_categories(df):
    print('\nInferring product categories from review text...')
    df = df.copy()
    df['category'] = df.apply(infer_category, axis=1)
    dist = df['category'].value_counts()
    print('  Category distribution:')
    for cat, n in dist.items():
        print(f'    {cat:<30}: {n:,} ({n/len(df)*100:.1f}%)')
    return df


def cross_category_cluster_analysis(df, cluster_col='cluster'):
    if 'category' not in df.columns:
        df = enrich_with_categories(df)
    cross = pd.crosstab(df[cluster_col], df['category'], normalize='index') * 100

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    fig.suptitle('Cross-Category Cluster Analysis', fontweight='bold', fontsize=14)

    sns.heatmap(cross, annot=True, fmt='.1f', cmap='Blues',
                linewidths=0.5, ax=axes[0], cbar_kws={'label': '% of cluster'})
    axes[0].set_title('Category % within Each Cluster', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)

    cross.plot(kind='bar', stacked=True, ax=axes[1], colormap='tab20',
               edgecolor='white', linewidth=0.5)
    axes[1].set_title('Category Composition per Cluster', fontweight='bold')
    axes[1].set_xlabel('Cluster'); axes[1].set_ylabel('% of reviews')
    axes[1].legend(loc='upper right', fontsize=7, bbox_to_anchor=(1.3, 1))
    axes[1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'category_cluster_analysis.png', dpi=130, bbox_inches='tight')
    plt.show()
    print('Saved: outputs/category_cluster_analysis.png')
    return cross
