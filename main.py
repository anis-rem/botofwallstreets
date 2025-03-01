import praw
import time
import os
import random
from dotenv import load_dotenv

load_dotenv()
reddit = praw.Reddit(
    client_id=os.getenv("PRAW_CLIENT_ID"),
    client_secret=os.getenv("PRAW_CLIENT_SECRET"),
    user_agent=os.getenv("PRAW_USER_AGENT")
)
try:
    print("Logged in as:", reddit.user.me())
except Exception as e:
    print("Error:", e)

assets = [
    # Major Stock Indices
    "^GSPC", "^DJI", "^IXIC", "^RUT", "^FTSE", "^N225", "^DAX", "^HSI",
    "^VIX", "^STOXX50E", "^AXJO", "^BVSP", "^KS11", "^MXX", "^TWII", "^SET.BK",

    # Big Tech Stocks (FAANG & More)
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "TSLA", "NFLX",
    "INTC", "IBM", "ORCL", "CSCO", "ADBE", "AVGO", "QCOM", "TXN", "CRM", "PYPL",

    # Financial & Banking Stocks
    "JPM", "GS", "BAC", "WFC", "C", "MS", "BLK", "AXP", "V", "MA", "SQ", "COF",
    "FIS", "DFS", "BRK.B", "BRK.A",

    # Popular Dividend & Blue-Chip Stocks
    "KO", "PEP", "JNJ", "PG", "MCD", "VZ", "T", "XOM", "CVX", "MO", "PFE", "MRK",
    "ABBV", "TGT", "HD", "LOW", "COST", "WMT", "NKE", "SBUX", "DIS", "UPS", "FDX",

    # Meme Stocks & High-Risk Picks
    "GME", "AMC", "BBBYQ", "BB", "PLTR", "SOFI", "SPCE", "LCID", "RIVN", "TSM",
    "ARKK", "FUBO", "NKLA", "DNA", "BABA", "JD", "PDD", "NIO", "LI",

    # Energy & EV Stocks
    "XOM", "CVX", "NIO", "PLUG", "F", "GM", "CHPT", "QS", "RUN", "SEDG", "ENPH",
    "BE", "FCEL", "HYLN", "NEE", "ED", "DUK", "DTE",

    # Crypto Stocks & ETFs
    "COIN", "MARA", "RIOT", "BITO", "HUT", "GBTC", "MSTR", "SI", "HIVE", "BKKT",

    # Popular Cryptocurrencies
    "BTC", "ETH", "BNB", "XRP", "DOGE", "SOL", "ADA", "MATIC", "DOT", "SHIB",
    "LTC", "AVAX", "ATOM", "LINK", "XLM", "ALGO", "UNI", "FTM", "ICP", "APE",

    # Commodity ETFs & Futures
    "GLD", "SLV", "USO", "UNG", "PALL", "DBC", "DBO", "WEAT", "CORN", "SOYB",
    "LIT", "REMX", "URA", "COPX", "SGOL", "IAU", "PLTM", "JJG",

    # Tech & AI Stocks
    "AI", "SNOW", "CRWD", "NET", "ZM", "DDOG", "MDB", "U", "ROKU", "TWLO",
    "TEAM", "ZS", "OKTA", "DOCU", "PATH", "BILL",

    # Miscellaneous & High-Growth Stocks
    "SHOP", "DKNG", "UBER", "LYFT", "ABNB", "RBLX", "PINS", "SNAP", "SPOT",
    "ASML", "SQM", "TDOC", "TTD", "WBD", "MTCH", "ETSY", "ZM", "BILL",

    # Bonds & Interest Rate ETFs
    "TLT", "IEF", "SHY", "BND", "LQD", "HYG", "AGG", "TIP", "MUB"
]

subreddits = [
    "trading", "stockmarket", "forex", "investing", "investor",
    "Etoro", "finance", "stocks", "Forex", "StockMarket",
    "wallstreetbets", "algotrading", "Daytrading", "CryptoCurrency",
    "CryptoMarkets", "Bitcoin", "ethereum", "dogecoin", "options",
    "SatoshiStreetBets", "DeFi", "CryptoTechnology"
]

post_limit = 50
content = []
timestart = time.time()
for index, subred in enumerate(subreddits, start=1):
    try:
        print(f"Processing subreddit {index}/{len(subreddits)}: r/{subred}")
        subreddit = reddit.subreddit(subred)
        post_count = 0

        for post in subreddit.new(limit=post_limit):
            post_count += 1
            post_content = post.selftext if post.selftext.strip() else "[No text]"
            content.append(post_content)
            if post_count % 20 == 0:
                print(f"  Processed {post_count} posts from r/{subred}")

        print(f"Completed r/{subred} - found {post_count} posts")

    except Exception as e:
        print(f"Error processing r/{subred}: {e}")
        if "429" in str(e):
            sleep_time = random.uniform(30, 60)
            print(f"Rate limited! Sleeping for {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)

print(f"\nTotal posts collected: {len(content)}")

timend = time.time()
elapsed_time = timend - timestart
print(f"It took {elapsed_time:.2f} seconds")
if elapsed_time > 0:
    posts_per_second = len(content) / elapsed_time
    print(f"Performance: {posts_per_second:.2f} posts/second")
print(content)