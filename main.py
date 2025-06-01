import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import warnings
from tqdm import tqdm
import time
import gc

warnings.filterwarnings('ignore')

# Ensure VADER lexicon is downloaded
nltk.download('vader_lexicon', quiet=True)

# Initialize VADER model
sia = SentimentIntensityAnalyzer()

print("Loading dataset...")
df = pd.read_csv('reddit_financial_posts.csv')
print(f"Loaded {len(df)} posts")


# Helper function to display progress with speed tracking
def process_with_progress(df, func, column_name, description):
    """Process dataframe with progress bar and speed tracking"""
    print(f"\n{description}")
    start_time = time.time()

    # Create progress bar with additional stats
    results = []
    with tqdm(total=len(df), desc=f"{description} Progress",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

        for idx, text in enumerate(df['tokenized_content']):
            result = func(text)
            results.append(result)

            # Update progress bar every 100 items or at the end
            if (idx + 1) % 100 == 0 or idx == len(df) - 1:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                pbar.set_postfix({
                    'Rate': f'{rate:.1f} items/sec',
                    'ETA': f'{(len(df) - idx - 1) / rate:.0f}s' if rate > 0 else 'N/A'
                })
                pbar.update(min(100, len(df) - pbar.n))

    end_time = time.time()
    total_time = end_time - start_time

    df[column_name] = results

    print(f"✅ {description} completed!")
    print(f"   Time: {total_time:.2f} seconds")
    print(f"   Rate: {len(df) / total_time:.1f} items/second")
    print(f"   Total items processed: {len(df)}")

    return total_time


# VADER sentiment classification (fastest option)
def classify_sentiment_vader(text):
    sentiment_score = sia.polarity_scores(str(text))['compound']
    if sentiment_score <= -0.05:
        return 'Don\'t Recommend'
    elif sentiment_score <= 0.5:
        return 'Recommend'
    else:
        return 'Highly Recommend'


# Process VADER first (very fast)
vader_time = process_with_progress(df, classify_sentiment_vader, 'Sentiment_Label_VADER', "VADER Sentiment Analysis")

# Show VADER results
print(f"\nVADER Sentiment Distribution:")
print(df['Sentiment_Label_VADER'].value_counts())

# Initialize timing dictionary
timing_results = {'VADER': vader_time}

continue_choice = input(
    "\nVADER analysis complete! Do you want to run the financial transformer models? (y/n): ").lower()

if continue_choice == 'y':
    try:
        from transformers import pipeline

        print("\n" + "=" * 60)
        print("LOADING FINANCIAL TRANSFORMER MODELS")
        print("=" * 60)

        # Model 1: Sigma Financial Sentiment (BEST for Reddit - 99.24% accuracy)
        print("🔄 Loading Sigma Financial Sentiment Model (Optimized for Reddit)...")
        sigma_pipeline = pipeline("sentiment-analysis",
                                  model="Sigma/financial-sentiment-analysis",
                                  max_length=256,
                                  truncation=True,
                                  device=-1,
                                  batch_size=1)
        print("✅ Sigma Financial Sentiment loaded successfully!")

        # Model 2: FinBERT (your existing one - keep it)
        print("🔄 Loading FinBERT...")
        finbert_pipeline = pipeline("sentiment-analysis",
                                    model="ProsusAI/finbert",
                                    tokenizer="ProsusAI/finbert",
                                    max_length=256,
                                    truncation=True,
                                    device=-1,
                                    batch_size=1)
        print("✅ FinBERT loaded successfully!")

        # Model 2: DistilRoBERTa Financial (much better than Twitter-RoBERTa)
        print("🔄 Loading DistilRoBERTa Financial News Model...")
        distil_financial_pipeline = pipeline("sentiment-analysis",
                                             model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
                                             max_length=256,
                                             truncation=True,
                                             device=-1,
                                             batch_size=1)
        print("✅ DistilRoBERTa Financial loaded successfully!")

        # Model 3: Financial RoBERTa Large (even more specialized)
        print("🔄 Loading Financial RoBERTa Large...")
        try:
            financial_roberta_pipeline = pipeline("sentiment-analysis",
                                                  model="soleimanian/financial-roberta-large-sentiment",
                                                  max_length=256,
                                                  truncation=True,
                                                  device=-1,
                                                  batch_size=1)
            use_roberta_large = True
            print("✅ Financial RoBERTa Large loaded successfully!")
        except Exception as e:
            print(f"❌ Financial RoBERTa Large failed to load: {e}")
            print("   Continuing without this model...")
            use_roberta_large = False

        print("\n" + "=" * 60)
        print("RUNNING SENTIMENT ANALYSIS")
        print("=" * 60)


        # Sigma Financial Sentiment classification (BEST for Reddit)
        def classify_sentiment_sigma(text):
            try:
                text_str = str(text)[:200]
                result = sigma_pipeline(text_str)[0]
                label = result['label'].lower()

                if 'negative' in label:
                    return 'Don\'t Recommend'
                elif 'neutral' in label:
                    return 'Recommend'
                else:  # positive
                    return 'Highly Recommend'
            except Exception as e:
                return 'Recommend'


        # FinBERT sentiment classification
        def classify_sentiment_finbert(text):
            try:
                text_str = str(text)[:200]
                result = finbert_pipeline(text_str)[0]
                label = result['label'].lower()

                if label == 'negative':
                    return 'Don\'t Recommend'
                elif label == 'neutral':
                    return 'Recommend'
                else:  # positive
                    return 'Highly Recommend'
            except Exception as e:
                return 'Recommend'


        # DistilRoBERTa Financial sentiment classification
        def classify_sentiment_distil_financial(text):
            try:
                text_str = str(text)[:200]
                result = distil_financial_pipeline(text_str)[0]
                label = result['label'].upper()

                # This model outputs: NEGATIVE, NEUTRAL, POSITIVE
                if label == 'NEGATIVE':
                    return 'Don\'t Recommend'
                elif label == 'NEUTRAL':
                    return 'Recommend'
                else:  # POSITIVE
                    return 'Highly Recommend'
            except Exception as e:
                return 'Recommend'


        # Financial RoBERTa Large sentiment classification
        def classify_sentiment_financial_roberta(text):
            try:
                text_str = str(text)[:200]
                result = financial_roberta_pipeline(text_str)[0]
                label = result['label'].upper()

                # Check what labels this model outputs and map accordingly
                if 'NEGATIVE' in label or 'BEARISH' in label:
                    return 'Don\'t Recommend'
                elif 'NEUTRAL' in label:
                    return 'Recommend'
                else:  # POSITIVE or BULLISH
                    return 'Highly Recommend'
            except Exception as e:
                return 'Recommend'


        # Process each model with enhanced progress tracking
        sigma_time = process_with_progress(df, classify_sentiment_sigma, 'Sentiment_Label_Sigma_Financial',
                                           "Sigma Financial Sentiment Analysis (Reddit-Optimized)")
        timing_results['Sigma_Financial'] = sigma_time

        # Free up memory
        print("🧹 Cleaning up Sigma model from memory...")
        del sigma_pipeline
        gc.collect()

        finbert_time = process_with_progress(df, classify_sentiment_finbert, 'Sentiment_Label_FinBERT',
                                             "FinBERT Sentiment Analysis")
        timing_results['FinBERT'] = finbert_time

        # Free up memory
        print("🧹 Cleaning up FinBERT from memory...")
        del finbert_pipeline
        gc.collect()

        distil_time = process_with_progress(df, classify_sentiment_distil_financial,
                                            'Sentiment_Label_DistilRoBERTa_Financial',
                                            "DistilRoBERTa Financial Analysis")
        timing_results['DistilRoBERTa_Financial'] = distil_time

        # Free up memory
        print("🧹 Cleaning up DistilRoBERTa from memory...")
        del distil_financial_pipeline
        gc.collect()

        # Process Financial RoBERTa Large if available
        if use_roberta_large:
            roberta_large_time = process_with_progress(df, classify_sentiment_financial_roberta,
                                                       'Sentiment_Label_Financial_RoBERTa_Large',
                                                       "Financial RoBERTa Large Analysis")
            timing_results['Financial_RoBERTa_Large'] = roberta_large_time

            print("🧹 Cleaning up Financial RoBERTa Large from memory...")
            del financial_roberta_pipeline
            gc.collect()
        else:
            roberta_large_time = 0


        # Create ensemble prediction (majority vote)
        def create_ensemble_prediction(row):
            votes = [
                row['Sentiment_Label_VADER'],
                row['Sentiment_Label_Sigma_Financial'],
                row['Sentiment_Label_FinBERT'],
                row['Sentiment_Label_DistilRoBERTa_Financial']
            ]

            if use_roberta_large:
                votes.append(row['Sentiment_Label_Financial_RoBERTa_Large'])

            # Count votes
            vote_counts = {}
            for vote in votes:
                vote_counts[vote] = vote_counts.get(vote, 0) + 1

            # Return majority vote (or first one if tie)
            return max(vote_counts.items(), key=lambda x: x[1])[0]


        print("\n🔄 Creating ensemble predictions...")
        start_time = time.time()

        with tqdm(total=len(df), desc="Ensemble Creation Progress") as pbar:
            ensemble_results = []
            for idx, row in df.iterrows():
                ensemble_results.append(create_ensemble_prediction(row))
                if (idx + 1) % 1000 == 0 or idx == len(df) - 1:
                    pbar.update(min(1000, len(df) - pbar.n))

        df['Sentiment_Label_Ensemble'] = ensemble_results
        ensemble_time = time.time() - start_time
        timing_results['Ensemble'] = ensemble_time

        print(f"✅ Ensemble creation completed in {ensemble_time:.2f} seconds!")

        print("\n" + "=" * 80)
        print("🎉 ALL FINANCIAL SENTIMENT ANALYSIS COMPLETED! 🎉")
        print("=" * 80)

        # Display timing results
        print("\n⏱️  PERFORMANCE SUMMARY:")
        print("-" * 50)
        total_time = sum(timing_results.values())
        for model, time_taken in timing_results.items():
            rate = len(df) / time_taken if time_taken > 0 else 0
            percentage = (time_taken / total_time) * 100 if total_time > 0 else 0
            print(f"{model:25} | {time_taken:8.2f}s | {rate:8.1f} items/sec | {percentage:5.1f}%")

        print(f"{'TOTAL TIME':25} | {total_time:8.2f}s | {len(df) / total_time:8.1f} items/sec | 100.0%")

        # Save final results with ALL columns
        output_filename = 'reddit_financial_posts_complete_analysis.csv'
        print(f"\n💾 Saving complete results to '{output_filename}'...")
        df.to_csv(output_filename, index=False)
        print("✅ All results saved successfully!")

        # Show detailed comparison
        print("\n" + "=" * 80)
        print("📊 SENTIMENT ANALYSIS COMPARISON:")
        print("=" * 80)

        models_to_show = [
            ('VADER (General Purpose)', 'Sentiment_Label_VADER'),
            ('Sigma Financial (Reddit-Optimized - 99.24% Accuracy)', 'Sentiment_Label_Sigma_Financial'),
            ('FinBERT (Financial Specialist)', 'Sentiment_Label_FinBERT'),
            ('DistilRoBERTa Financial (News Specialist)', 'Sentiment_Label_DistilRoBERTa_Financial'),
        ]

        if use_roberta_large:
            models_to_show.append(('Financial RoBERTa Large (Advanced)', 'Sentiment_Label_Financial_RoBERTa_Large'))

        models_to_show.append(('ENSEMBLE (Majority Vote - RECOMMENDED)', 'Sentiment_Label_Ensemble'))

        for model_name, column_name in models_to_show:
            print(f"\n{model_name}:")
            counts = df[column_name].value_counts()
            for sentiment, count in counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {sentiment:18} | {count:6d} ({percentage:5.1f}%)")

        # Show sample predictions for verification
        print("\n" + "=" * 80)
        print("🔍 SAMPLE PREDICTIONS (First 5 posts):")
        print("=" * 80)

        sample_cols = ['tokenized_content', 'Sentiment_Label_VADER', 'Sentiment_Label_Sigma_Financial',
                       'Sentiment_Label_FinBERT',
                       'Sentiment_Label_DistilRoBERTa_Financial']
        if use_roberta_large:
            sample_cols.append('Sentiment_Label_Financial_RoBERTa_Large')
        sample_cols.append('Sentiment_Label_Ensemble')

        for i in range(min(5, len(df))):
            print(f"\n📝 Post {i + 1}: {df.iloc[i]['tokenized_content'][:100]}...")
            print(f"   VADER:              {df.iloc[i]['Sentiment_Label_VADER']}")
            print(f"   🏆 SIGMA (Reddit):   {df.iloc[i]['Sentiment_Label_Sigma_Financial']}")
            print(f"   FinBERT:            {df.iloc[i]['Sentiment_Label_FinBERT']}")
            print(f"   DistilRoBERTa:      {df.iloc[i]['Sentiment_Label_DistilRoBERTa_Financial']}")
            if use_roberta_large:
                print(f"   RoBERTa Large:      {df.iloc[i]['Sentiment_Label_Financial_RoBERTa_Large']}")
            print(f"   🎯 ENSEMBLE:        {df.iloc[i]['Sentiment_Label_Ensemble']}")
            print("-" * 80)

        print(f"\n📋 FINAL CSV CONTAINS {len(df.columns)} COLUMNS:")
        sentiment_columns = [col for col in df.columns if 'Sentiment_Label' in col]
        print("Original columns + New sentiment columns:")
        for col in sentiment_columns:
            print(f"  • {col}")

    except ImportError as e:
        print(f"❌ Transformers not available: {e}")
        print("Only VADER analysis completed.")
        # Save VADER-only results
        output_filename = 'reddit_financial_posts_vader_only.csv'
        df.to_csv(output_filename, index=False)
        print(f"💾 VADER results saved to '{output_filename}'")

else:
    print("⏭️  Skipping transformer models. VADER results are ready to use!")
    output_filename = 'reddit_financial_posts_vader_only.csv'
    df.to_csv(output_filename, index=False)
    print(f"💾 File saved as: '{output_filename}'")

print(f"\n🎉 ANALYSIS COMPLETE! 🎉")
print("📈 Recommended model to use: ENSEMBLE (combines all models for best accuracy)")
print(f"📁 Check your output CSV file for all sentiment labels in separate columns!")