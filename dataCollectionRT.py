import os
import time
import re
import random
import nltk
import praw
import csv
import threading
from datetime import datetime
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
except:
    pass
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
load_dotenv()
reddit = praw.Reddit(
    client_id=os.getenv("PRAW_CLIENT_ID"),
    client_secret=os.getenv("PRAW_CLIENT_SECRET"),
    user_agent=os.getenv("PRAW_USER_AGENT")
)

assets = [
    # Major Stock Indices
    "^GSPC", "^DJI", "^IXIC", "^VIX",

    # Big Tech & AI Stocks
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",

    # Financial Giants
    "JPM", "GS", "BAC", "BRK.B",

    # Popular Dividend & Blue-Chip Stocks
    "JNJ", "PG", "KO", "XOM", "PEP", "CVX",

    # Cryptocurrencies & Crypto Stocks
    "BTC", "ETH", "MARA", "RIOT",

    # Commodity ETFs
    "GLD", "SLV"
]


# Subreddits to scrape
subreddits = [
    "trading", "stockmarket", "forex", "investing", "investor",
    "Etoro", "finance", "stocks", "Forex", "StockMarket",
    "wallstreetbets", "algotrading", "Daytrading", "CryptoCurrency",
    "CryptoMarkets", "Bitcoin", "ethereum", "dogecoin", "options",
    "SatoshiStreetBets", "DeFi", "CryptoTechnology"
]

# Configuration
CSV_FILENAME = "reddit_financial_posts.csv"
COLLECTION_INTERVAL_MINUTES = 5  # How often to collect data
POSTS_PER_BATCH = 30  # How many posts to collect per subreddit per batch
collected_post_ids = set()


def load_existing_post_ids(filename):
    """Load existing post IDs from CSV to avoid duplicates"""
    global collected_post_ids
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if 'post_id' in row:
                    collected_post_ids.add(row['post_id'])
        print(f"✅ Loaded {len(collected_post_ids)} existing post IDs from {filename}")
    except FileNotFoundError:
        print(f"📝 No existing file found. Starting fresh collection.")
    except Exception as e:
        print(f"❌ Error loading existing post IDs: {e}")


def detect_assets_in_text(text, assets_list):
    """Detect mentioned assets in text"""
    text_upper = text.upper()
    detected = []
    for asset in assets_list:
        if asset.upper() in text_upper:
            detected.append(asset)
    return detected


def save_posts_to_csv(posts_data, filename):
    """Save posts data to CSV file"""
    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['collection_timestamp', 'post_timestamp', 'subreddit', 'post_id',
                      'title', 'content', 'cleaned_content', 'tokenized_content',
                      'detected_assets', 'score', 'num_comments', 'post_url', 'author']

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if file is new
        if not file_exists:
            writer.writeheader()
            print(f"📄 Created new CSV file: {filename}")

        # Write data
        new_posts_count = 0
        for post in posts_data:
            if post['post_id'] not in collected_post_ids:
                writer.writerow(post)
                collected_post_ids.add(post['post_id'])
                new_posts_count += 1

        if new_posts_count > 0:
            print(f"💾 Added {new_posts_count} new posts to {filename}")
        else:
            print("⚡ No new posts to add (all were duplicates)")


def collect_posts(limit_per_sub=120):
    """Collect posts from subreddits and return processed data"""
    posts_data = []
    collection_time = datetime.now()

    print(
        f"\n🔍 [{collection_time.strftime('%Y-%m-%d %H:%M:%S')}] Starting collection from {len(subreddits)} subreddits...")

    for subreddit_name in subreddits:
        try:
            print(f"  📊 Processing r/{subreddit_name}...")
            subreddit = reddit.subreddit(subreddit_name)
            new_posts = 0

            for post in subreddit.new(limit=limit_per_sub):
                try:
                    # Skip if we already have this post
                    if post.id in collected_post_ids:
                        continue

                    # Get post content
                    content = post.selftext if post.selftext.strip() else "[No text content]"
                    title = post.title

                    # Clean content
                    cleaned_content = re.sub(r"http\S+|www\S+", "", content)
                    cleaned_content = re.sub(r'[^\s\w$%+\-.]', '', cleaned_content)
                    cleaned_content = re.sub(r"\s+", " ", cleaned_content).strip()

                    # Tokenize content
                    try:
                        tokens = nltk.word_tokenize(cleaned_content)
                        tokens = [lemmatizer.lemmatize(word) for word in tokens
                                  if word.lower() not in stop_words and len(word) > 1]
                        tokenized_content = ' '.join(tokens)
                    except:
                        tokenized_content = cleaned_content

                    # Detect assets
                    full_text = f"{title} {content}"
                    detected_assets = detect_assets_in_text(full_text, assets)

                    # Create post data
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
                    new_posts += 1

                except Exception as e:
                    continue

            print(f"    ✅ Collected {new_posts} new posts from r/{subreddit_name}")

        except Exception as e:
            print(f"  ❌ Error processing r/{subreddit_name}: {e}")
            if "429" in str(e) or "rate" in str(e).lower():
                sleep_time = random.uniform(60, 120)
                print(f"    ⏳ Rate limited! Sleeping for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)

    return posts_data


def continuous_collection():
    """Main continuous collection loop"""
    print("🚀 REDDIT FINANCIAL DATA COLLECTOR STARTED")
    print("=" * 60)
    print(f"📁 Data will be saved to: {CSV_FILENAME}")
    print(f"⏰ Collection interval: {COLLECTION_INTERVAL_MINUTES} minutes")
    print(f"📊 Posts per batch: {POSTS_PER_BATCH} per subreddit")
    print(f"🎯 Monitoring {len(subreddits)} subreddits")
    print(f"💰 Tracking {len(assets)} financial assets")
    print("=" * 60)
    print("Press Ctrl+C to stop collection")
    print("=" * 60)

    # Load existing post IDs to avoid duplicates
    load_existing_post_ids(CSV_FILENAME)

    batch_number = 1

    try:
        while True:
            print(f"\n🔄 BATCH {batch_number} STARTED")
            start_time = time.time()

            # Collect posts
            posts_data = collect_posts(POSTS_PER_BATCH)

            if posts_data:
                # Save to CSV
                save_posts_to_csv(posts_data, CSV_FILENAME)

                # Statistics
                total_collected = len(posts_data)
                posts_with_assets = sum(1 for post in posts_data if post['detected_assets'])
                collection_time = time.time() - start_time

                print(f"\n📈 BATCH {batch_number} SUMMARY:")
                print(f"  📊 New posts collected: {total_collected}")
                print(f"  💰 Posts with assets: {posts_with_assets}")
                print(f"  🗄️  Total unique posts in database: {len(collected_post_ids)}")
                print(f"  ⏱️  Collection time: {collection_time:.2f} seconds")

                if posts_with_assets > 0:
                    # Show top assets mentioned
                    asset_counts = {}
                    for post in posts_data:
                        if post['detected_assets']:
                            for asset in post['detected_assets'].split(','):
                                asset = asset.strip()
                                if asset:
                                    asset_counts[asset] = asset_counts.get(asset, 0) + 1

                    if asset_counts:
                        top_assets = sorted(asset_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                        print(
                            f"  🔥 Top mentioned assets: {', '.join([f'{asset}({count})' for asset, count in top_assets])}")
            else:
                print(f"\n📭 BATCH {batch_number}: No new posts found")

            batch_number += 1

            # Wait for next collectio
            print(f"\n⏳ Waiting {COLLECTION_INTERVAL_MINUTES} minutes until next collection...")
            print("   (Press Ctrl+C to stop)")
            time.sleep(COLLECTION_INTERVAL_MINUTES * 60)

    except KeyboardInterrupt:
        print("\n\n🛑 Collection stopped by user!")
        print(f"📊 Final Statistics:")
        print(f"  - Total batches processed: {batch_number - 1}")
        print(f"  - Total unique posts collected: {len(collected_post_ids)}")
        print(f"  - Data saved in: {CSV_FILENAME}")
        print("👋 Thank you for using Reddit Financial Data Collector!")



continuous_collection()