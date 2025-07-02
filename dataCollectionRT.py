import os
import time
import re
import random
import nltk
import praw
import csv
from datetime import datetime
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("PRAW_CLIENT_ID"),
    client_secret=os.getenv("PRAW_CLIENT_SECRET"),
    user_agent=os.getenv("PRAW_USER_AGENT")
)

assets = [
    "^GSPC", "^DJI", "^IXIC", "^VIX",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "JPM", "GS", "BAC", "BRK.B",
    "JNJ", "PG", "KO", "XOM", "PEP", "CVX",
    "BTC", "ETH", "MARA", "RIOT",
    "GLD", "SLV"
]

subreddits = [
    "trading", "stockmarket", "forex", "investing", "investor",
    "Etoro", "finance", "stocks", "Forex", "StockMarket",
    "wallstreetbets", "algotrading", "Daytrading", "CryptoCurrency",
    "CryptoMarkets", "Bitcoin", "ethereum", "dogecoin", "options",
    "SatoshiStreetBets", "DeFi", "CryptoTechnology"
]

CSV_FILENAME = "reddit_financial_posts.csv"
COLLECTION_INTERVAL_MINUTES = 5
POSTS_PER_BATCH = 30
collected_post_ids = set()

def load_existing_post_ids(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if 'post_id' in row:
                    collected_post_ids.add(row['post_id'])
    except:
        pass

def detect_assets_in_text(text, assets_list):
    text_upper = text.upper()
    return [asset for asset in assets_list if asset.upper() in text_upper]

def save_posts_to_csv(posts_data, filename):
    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['collection_timestamp', 'post_timestamp', 'subreddit', 'post_id',
                      'title', 'content', 'cleaned_content', 'tokenized_content',
                      'detected_assets', 'score', 'num_comments', 'post_url', 'author']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for post in posts_data:
            if post['post_id'] not in collected_post_ids:
                writer.writerow(post)
                collected_post_ids.add(post['post_id'])

def collect_posts(limit_per_sub=120):
    posts_data = []
    collection_time = datetime.now()

    for subreddit_name in subreddits:
        try:
            subreddit = reddit.subreddit(subreddit_name)
            for post in subreddit.new(limit=limit_per_sub):
                if post.id in collected_post_ids:
                    continue

                content = post.selftext if post.selftext.strip() else "[No text content]"
                title = post.title

                cleaned_content = re.sub(r"http\S+|www\S+", "", content)
                cleaned_content = re.sub(r'[^\s\w$%+\-.]', '', cleaned_content)
                cleaned_content = re.sub(r"\s+", " ", cleaned_content).strip()

                try:
                    tokens = nltk.word_tokenize(cleaned_content)
                    tokens = [lemmatizer.lemmatize(word) for word in tokens
                              if word.lower() not in stop_words and len(word) > 1]
                    tokenized_content = ' '.join(tokens)
                except:
                    tokenized_content = cleaned_content

                full_text = f"{title} {content}"
                detected_assets = detect_assets_in_text(full_text, assets)

                post_data = {
                    'collection_timestamp': collection_time.isoformat(),
                    'post_timestamp': datetime.fromtimestamp(post.created_utc).isoformat(),
                    'subreddit': subreddit_name,
                    'post_id': post.id,
                    'title': title,
                    'content': content,
                    'cleaned_content': cleaned_content,
                    'tokenized_content': tokenized_content,
                    'detected_assets': ','.join(detected_assets),
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'post_url': f"https://reddit.com{post.permalink}",
                    'author': str(post.author) if post.author else '[deleted]'
                }

                posts_data.append(post_data)
        except:
            time.sleep(random.uniform(60, 120))
    return posts_data

def continuous_collection():
    load_existing_post_ids(CSV_FILENAME)
    batch_number = 1

    while True:
        start_time = time.time()
        posts_data = collect_posts(POSTS_PER_BATCH)

        if posts_data:
            save_posts_to_csv(posts_data, CSV_FILENAME)
            total_collected = len(posts_data)
            posts_with_assets = sum(1 for post in posts_data if post['detected_assets'])

            asset_counts = {}
            for post in posts_data:
                if post['detected_assets']:
                    for asset in post['detected_assets'].split(','):
                        asset = asset.strip()
                        if asset:
                            asset_counts[asset] = asset_counts.get(asset, 0) + 1

            top_assets = sorted(asset_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"Batch {batch_number} | New: {total_collected} | With Assets: {posts_with_assets} | Top: {top_assets}")
        else:
            print(f"Batch {batch_number}: No new posts")

        batch_number += 1
        time.sleep(COLLECTION_INTERVAL_MINUTES * 60)

continuous_collection()