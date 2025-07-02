import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import warnings
from tqdm import tqdm
import time
import gc

warnings.filterwarnings('ignore')

nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()

print("Loading dataset...")
df = pd.read_csv('reddit_financial_posts.csv')
print(f"Loaded {len(df)} posts")


def process_with_progress(df, func, column_name, description):
    print(f"\n{description}")
    start_time = time.time()

    results = []
    with tqdm(total=len(df), desc=f"{description} Progress",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

        for idx, text in enumerate(df['tokenized_content']):
            result = func(text)
            results.append(result)

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

    print(f"Completed in {total_time:.2f} seconds")
    print(f"Rate: {len(df) / total_time:.1f} items/second")

    return total_time


def classify_sentiment_vader(text):
    sentiment_score = sia.polarity_scores(str(text))['compound']
    if sentiment_score <= -0.05:
        return 'Don\'t Recommend'
    elif sentiment_score <= 0.5:
        return 'Recommend'
    else:
        return 'Highly Recommend'


vader_time = process_with_progress(df, classify_sentiment_vader, 'Sentiment_Label_VADER', "VADER Sentiment Analysis")

print(f"\nVADER Sentiment Distribution:")
print(df['Sentiment_Label_VADER'].value_counts())

timing_results = {'VADER': vader_time}

continue_choice = input("\nVADER analysis complete. Run financial transformer models? (y/n): ").lower()

if continue_choice == 'y':
    try:
        from transformers import pipeline

        print("\nLoading financial transformer models...")

        print("Loading Sigma Financial Sentiment Model...")
        sigma_pipeline = pipeline("sentiment-analysis",
                                  model="Sigma/financial-sentiment-analysis",
                                  max_length=256,
                                  truncation=True,
                                  device=-1,
                                  batch_size=1)

        print("Loading FinBERT...")
        finbert_pipeline = pipeline("sentiment-analysis",
                                    model="ProsusAI/finbert",
                                    tokenizer="ProsusAI/finbert",
                                    max_length=256,
                                    truncation=True,
                                    device=-1,
                                    batch_size=1)

        print("Loading DistilRoBERTa Financial News Model...")
        distil_financial_pipeline = pipeline("sentiment-analysis",
                                             model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
                                             max_length=256,
                                             truncation=True,
                                             device=-1,
                                             batch_size=1)

        print("Loading Financial RoBERTa Large...")
        try:
            financial_roberta_pipeline = pipeline("sentiment-analysis",
                                                  model="soleimanian/financial-roberta-large-sentiment",
                                                  max_length=256,
                                                  truncation=True,
                                                  device=-1,
                                                  batch_size=1)
            use_roberta_large = True
        except Exception as e:
            print(f"Financial RoBERTa Large failed to load: {e}")
            print("Continuing without this model...")
            use_roberta_large = False


        def classify_sentiment_sigma(text):
            try:
                text_str = str(text)[:200]
                result = sigma_pipeline(text_str)[0]
                label = result['label'].lower()

                if 'negative' in label:
                    return 'Don\'t Recommend'
                elif 'neutral' in label:
                    return 'Recommend'
                else:
                    return 'Highly Recommend'
            except Exception as e:
                return 'Recommend'


        def classify_sentiment_finbert(text):
            try:
                text_str = str(text)[:200]
                result = finbert_pipeline(text_str)[0]
                label = result['label'].lower()

                if label == 'negative':
                    return 'Don\'t Recommend'
                elif label == 'neutral':
                    return 'Recommend'
                else:
                    return 'Highly Recommend'
            except Exception as e:
                return 'Recommend'


        def classify_sentiment_distil_financial(text):
            try:
                text_str = str(text)[:200]
                result = distil_financial_pipeline(text_str)[0]
                label = result['label'].upper()

                if label == 'NEGATIVE':
                    return 'Don\'t Recommend'
                elif label == 'NEUTRAL':
                    return 'Recommend'
                else:
                    return 'Highly Recommend'
            except Exception as e:
                return 'Recommend'


        def classify_sentiment_financial_roberta(text):
            try:
                text_str = str(text)[:200]
                result = financial_roberta_pipeline(text_str)[0]
                label = result['label'].upper()

                if 'NEGATIVE' in label or 'BEARISH' in label:
                    return 'Don\'t Recommend'
                elif 'NEUTRAL' in label:
                    return 'Recommend'
                else:
                    return 'Highly Recommend'
            except Exception as e:
                return 'Recommend'


        sigma_time = process_with_progress(df, classify_sentiment_sigma, 'Sentiment_Label_Sigma_Financial',
                                           "Sigma Financial Sentiment Analysis")
        timing_results['Sigma_Financial'] = sigma_time

        del sigma_pipeline
        gc.collect()

        finbert_time = process_with_progress(df, classify_sentiment_finbert, 'Sentiment_Label_FinBERT',
                                             "FinBERT Sentiment Analysis")
        timing_results['FinBERT'] = finbert_time

        del finbert_pipeline
        gc.collect()

        distil_time = process_with_progress(df, classify_sentiment_distil_financial,
                                            'Sentiment_Label_DistilRoBERTa_Financial',
                                            "DistilRoBERTa Financial Analysis")
        timing_results['DistilRoBERTa_Financial'] = distil_time

        del distil_financial_pipeline
        gc.collect()

        if use_roberta_large:
            roberta_large_time = process_with_progress(df, classify_sentiment_financial_roberta,
                                                       'Sentiment_Label_Financial_RoBERTa_Large',
                                                       "Financial RoBERTa Large Analysis")
            timing_results['Financial_RoBERTa_Large'] = roberta_large_time

            del financial_roberta_pipeline
            gc.collect()


        def create_ensemble_prediction(row):
            votes = [
                row['Sentiment_Label_VADER'],
                row['Sentiment_Label_Sigma_Financial'],
                row['Sentiment_Label_FinBERT'],
                row['Sentiment_Label_DistilRoBERTa_Financial']
            ]

            if use_roberta_large:
                votes.append(row['Sentiment_Label_Financial_RoBERTa_Large'])

            vote_counts = {}
            for vote in votes:
                vote_counts[vote] = vote_counts.get(vote, 0) + 1

            return max(vote_counts.items(), key=lambda x: x[1])[0]


        print("\nCreating ensemble predictions...")
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
        total_time = sum(timing_results.values())
        for model, time_taken in timing_results.items():
            rate = len(df) / time_taken if time_taken > 0 else 0
            percentage = (time_taken / total_time) * 100 if total_time > 0 else 0
            print(f"{model:25} | {time_taken:8.2f}s | {rate:8.1f} items/sec | {percentage:5.1f}%")

        print(f"{'Total time':25} | {total_time:8.2f}s | {len(df) / total_time:8.1f} items/sec | 100.0%")

        output_filename = 'reddit_financial_posts_complete_analysis.csv'
        print(f"\nSaving complete results to '{output_filename}'...")
        df.to_csv(output_filename, index=False)
        print("All results saved successfully")

        print("\nSentiment analysis comparison:")

        models_to_show = [
            ('VADER (General Purpose)', 'Sentiment_Label_VADER'),
            ('Sigma Financial (Reddit-Optimized)', 'Sentiment_Label_Sigma_Financial'),
            ('FinBERT (Financial Specialist)', 'Sentiment_Label_FinBERT'),
            ('DistilRoBERTa Financial (News Specialist)', 'Sentiment_Label_DistilRoBERTa_Financial'),
        ]

        if use_roberta_large:
            models_to_show.append(('Financial RoBERTa Large (Advanced)', 'Sentiment_Label_Financial_RoBERTa_Large'))

        models_to_show.append(('Ensemble (Majority Vote - Recommended)', 'Sentiment_Label_Ensemble'))

        for model_name, column_name in models_to_show:
            print(f"\n{model_name}:")
            counts = df[column_name].value_counts()
            for sentiment, count in counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {sentiment:18} | {count:6d} ({percentage:5.1f}%)")

        print("\nSample predictions (first 5 posts):")

        for i in range(min(5, len(df))):
            print(f"\nPost {i + 1}: {df.iloc[i]['tokenized_content'][:100]}...")
            print(f"   VADER:              {df.iloc[i]['Sentiment_Label_VADER']}")
            print(f"   Sigma (Reddit):     {df.iloc[i]['Sentiment_Label_Sigma_Financial']}")
            print(f"   FinBERT:            {df.iloc[i]['Sentiment_Label_FinBERT']}")
            print(f"   DistilRoBERTa:      {df.iloc[i]['Sentiment_Label_DistilRoBERTa_Financial']}")
            if use_roberta_large:
                print(f"   RoBERTa Large:      {df.iloc[i]['Sentiment_Label_Financial_RoBERTa_Large']}")
            print(f"   Ensemble:           {df.iloc[i]['Sentiment_Label_Ensemble']}")

        sentiment_columns = [col for col in df.columns if 'Sentiment_Label' in col]
        print(f"\nFinal CSV contains {len(df.columns)} columns:")
        for col in sentiment_columns:
            print(f"  • {col}")

    except ImportError as e:
        print(f"Transformers not available: {e}")
        print("Only VADER analysis completed.")
        output_filename = 'reddit_financial_posts_vader_only.csv'
        df.to_csv(output_filename, index=False)
        print(f"VADER results saved to '{output_filename}'")

else:
    print("Skipping transformer models. VADER results are ready to use!")
    output_filename = 'reddit_financial_posts_vader_only.csv'
    df.to_csv(output_filename, index=False)
    print(f"File saved as: '{output_filename}'")

print(f"\nAnalysis complete!")