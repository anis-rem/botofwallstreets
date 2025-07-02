import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

df = pd.read_csv('reddit_financial_posts_classified_balanced.csv')

print(f"Dataset shape: {df.shape}")

sentiment_columns = [
    'sentiment_label',
    'Sentiment_Label_VADER',
    'Sentiment_Label_Sigma_Financial',
    'Sentiment_Label_FinBERT',
    'Sentiment_Label_DistilRoBERTa_Financial',
    'Sentiment_Label_Financial_RoBERTa_Large',
    'Sentiment_Label_Ensemble'
]

clean_names = {
    'sentiment_label': 'Bot of wallstreets',
    'Sentiment_Label_VADER': 'VADER',
    'Sentiment_Label_Sigma_Financial': 'Sigma financial',
    'Sentiment_Label_FinBERT': 'FinBERT',
    'Sentiment_Label_DistilRoBERTa_Financial': 'DistilRoBERTa',
    'Sentiment_Label_Financial_RoBERTa_Large': 'RoBERTa large',
    'Sentiment_Label_Ensemble': 'Ensemble'
}

existing_sentiment_cols = [col for col in sentiment_columns if col in df.columns]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

for i, col in enumerate(existing_sentiment_cols):
    if i < len(axes):
        sentiment_counts = df[col].value_counts()
        axes[i].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
        axes[i].set_title(f'{clean_names.get(col, col)} distribution', fontsize=12, fontweight='bold')

for i in range(len(existing_sentiment_cols), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.suptitle('Sentiment distribution across different methods', fontsize=16, fontweight='bold', y=1.02)
plt.show()

if 'Sentiment_Label_Ensemble' in df.columns and 'sentiment_label' in df.columns:
    valid_categories = ['Recommend', 'Highly Recommend', "Don't Recommend"]

    clean_df = df[['sentiment_label', 'Sentiment_Label_Ensemble']].dropna()

    mask = (clean_df['sentiment_label'].isin(valid_categories)) & (
        clean_df['Sentiment_Label_Ensemble'].isin(valid_categories))
    clean_df = clean_df[mask]

    bot_predictions = clean_df['sentiment_label']
    ensemble_predictions = clean_df['Sentiment_Label_Ensemble']

    accuracy = accuracy_score(ensemble_predictions, bot_predictions)
    print(f"Bot of wallstreets accuracy vs ensemble: {accuracy:.2%}")
    cm = confusion_matrix(ensemble_predictions, bot_predictions, labels=valid_categories)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=valid_categories, yticklabels=valid_categories)
    plt.title('Confusion matrix: Bot of wallstreets vs ensemble model', fontsize=16, fontweight='bold')
    plt.xlabel('Bot of wallstreets predictions', fontsize=14)
    plt.ylabel('Ensemble model', fontsize=14)
    plt.tight_layout()
    plt.show()

if 'Sentiment_Label_Ensemble' in df.columns:
    print("\nModel Accuracy Comparison vs Ensemble:")
    print("-" * 50)

    results = []
    for col in existing_sentiment_cols:
        if col != 'Sentiment_Label_Ensemble':

            clean_df = df[['Sentiment_Label_Ensemble', col]].dropna()
            valid_categories = ['Recommend', 'Highly Recommend', "Don't Recommend"]
            mask = (clean_df['Sentiment_Label_Ensemble'].isin(valid_categories)) & (
                clean_df[col].isin(valid_categories))
            clean_df = clean_df[mask]

            ensemble_predictions = clean_df['Sentiment_Label_Ensemble']
            model_predictions = clean_df[col]

            accuracy = accuracy_score(ensemble_predictions, model_predictions)
            results.append({
                'Model': clean_names.get(col, col),
                'Accuracy': accuracy,
                'Count': len(clean_df)
            })

    results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
    print(results_df.to_string(index=False))

    plt.figure(figsize=(12, 6))
    bars = plt.barh(results_df['Model'], results_df['Accuracy'], color='skyblue')
    plt.title('Model Accuracy vs Ensemble', fontsize=16, fontweight='bold')
    plt.xlabel('Accuracy Score', fontsize=12)
    plt.xlim(0, 1)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{width:.2%}', va='center')

    plt.tight_layout()
    plt.show()
    print("-" * 50)

print("\nAsset recommendation scoring system")
print("Scoring: Highly recommended = +2, Recommended = +1, Don't recommend = -2")


def parse_detected_assets(assets_str):
    if pd.isna(assets_str) or assets_str == '':
        return []

    assets_str = str(assets_str).strip()
    assets_str = assets_str.replace('[', '').replace(']', '').replace("'", '').replace('"', '')

    if ',' in assets_str:
        assets = [asset.strip() for asset in assets_str.split(',')]
    else:
        assets = [assets_str] if assets_str else []

    assets = [asset for asset in assets if asset and asset.strip()]
    return assets


df['parsed_assets'] = df['detected_assets'].apply(parse_detected_assets)

asset_scores = defaultdict(int)
asset_counts = defaultdict(int)
asset_sentiment_breakdown = defaultdict(lambda: {'Highly recommended': 0, 'Recommended': 0, "Don't recommend": 0})


def get_sentiment_score(sentiment):
    if pd.isna(sentiment):
        return 0, 'Other'

    sentiment_str = str(sentiment).lower().strip()

    if 'highly recommend' in sentiment_str:
        return 2, 'Highly recommended'
    elif "don't recommend" in sentiment_str or "dont recommend" in sentiment_str:
        return -2, "Don't recommend"
    elif sentiment_str == 'recommend' or 'recommend' in sentiment_str:
        return 1, 'Recommended'
    else:
        return 0, 'Other'

bot_sentiment_col = 'sentiment_label'
if 'Sentiment_Label_Ensemble' in df.columns:
    print("\nPrecision, Recall, and F1 Score Comparison (vs Ensemble):")
    print("-" * 60)

    summary = []
    valid_categories = ['Recommend', 'Highly Recommend', "Don't Recommend"]

    for col in existing_sentiment_cols:
        if col != 'Sentiment_Label_Ensemble':
            # Filter data for each model comparison
            clean_df = df[['Sentiment_Label_Ensemble', col]].dropna()

            # Filter to only include valid categories
            mask = (clean_df['Sentiment_Label_Ensemble'].isin(valid_categories)) & (
                clean_df[col].isin(valid_categories))
            clean_df = clean_df[mask]

            y_true = clean_df['Sentiment_Label_Ensemble']
            y_pred = clean_df[col]

            print(f"\nModel: {clean_names.get(col, col)}")
            print(classification_report(y_true, y_pred, zero_division=0))

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0
            )
            summary.append({
                'Model': clean_names.get(col, col),
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            })
    summary_df = pd.DataFrame(summary).sort_values(by='F1 Score', ascending=False)
    print("\nMacro-Averaged Scores Summary:")
    print(summary_df.to_string(index=False))

if bot_sentiment_col in df.columns:
    for idx, row in df.iterrows():
        sentiment = row[bot_sentiment_col]
        assets = row['parsed_assets']

        if assets and not pd.isna(sentiment):
            score, sentiment_category = get_sentiment_score(sentiment)

            if score != 0:
                for asset in assets:
                    asset_scores[asset] += score
                    asset_counts[asset] += 1
                    asset_sentiment_breakdown[asset][sentiment_category] += 1

    asset_rankings = []
    for asset in asset_scores:
        total_score = asset_scores[asset]
        count = asset_counts[asset]
        avg_score = total_score / count if count > 0 else 0
        asset_rankings.append((asset, total_score, count, avg_score))

    asset_rankings.sort(key=lambda x: x[1], reverse=True)

    print(f"\nAsset recommendation rankings")
    print(f"{'Rank':<5} {'Asset':<10} {'Total score':<12} {'Mentions':<10} {'Avg score':<10} {'Status'}")
    print("-" * 70)

    for i, (asset, total_score, count, avg_score) in enumerate(asset_rankings[:20], 1):
        if total_score > 0:
            status = "Recommended ✓"
        elif total_score < 0:
            status = "Don't recommend ✗"

        print(f"{i:<5} {asset:<10} {total_score:<12} {count:<10} {avg_score:<10.2f} {status}")

    if asset_rankings:
        top_assets = asset_rankings[:15]
        assets_names = [x[0] for x in top_assets]
        total_scores = [x[1] for x in top_assets]

        plt.figure(figsize=(14, 8))
        colors = ['green' if score > 0 else 'red' if score < 0 else 'gray' for score in total_scores]
        bars = plt.bar(assets_names, total_scores, color=colors, alpha=0.7, edgecolor='black')

        plt.title('Top assets by recommendation score', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Assets', fontsize=12)
        plt.ylabel('Total recommendation score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        for bar, score in zip(bars, total_scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + (0.1 if height >= 0 else -0.3),
                     f'{score}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')

        plt.tight_layout()
        plt.show()

        mention_counts = [x[2] for x in top_assets]
        plt.figure(figsize=(14, 6))
        plt.bar(assets_names, mention_counts, color='skyblue', edgecolor='navy', alpha=0.7)
        plt.title('Asset mention frequency', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Assets', fontsize=12)
        plt.ylabel('Number of mentions', fontsize=12)
        plt.xticks(rotation=45, ha='right')

        for i, count in enumerate(mention_counts):
            plt.text(i, count + 0.1, str(count), ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(figsize=(14, 8))

        highly_rec = [asset_sentiment_breakdown[asset]['Highly recommended'] for asset in assets_names]
        recommended = [asset_sentiment_breakdown[asset]['Recommended'] for asset in assets_names]
        not_recommended = [asset_sentiment_breakdown[asset]["Don't recommend"] for asset in assets_names]

        ax.bar(assets_names, highly_rec, label='Highly recommended (+2)', color='darkgreen', alpha=0.8)
        ax.bar(assets_names, recommended, bottom=highly_rec, label='Recommended (+1)', color='lightgreen', alpha=0.8)
        ax.bar(assets_names, not_recommended, bottom=[h + r for h, r in zip(highly_rec, recommended)],
               label="Don't recommend (-2)", color='red', alpha=0.8)

        ax.set_title('Sentiment breakdown by asset', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Assets', fontsize=12)
        ax.set_ylabel('Number of mentions', fontsize=12)
        ax.legend()
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()