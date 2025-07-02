import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_and_tokenize(text):
    try:
        text = re.sub(r"http\S+|www\S+", "", str(text))
        text = re.sub(r'[^\s\w$%+\-.]', '', text)
        text = re.sub(r"\s+", " ", text).strip()

        tokens = nltk.word_tokenize(text)
        tokens = [
            lemmatizer.lemmatize(word.lower()) for word in tokens
            if word.lower() not in stop_words and len(word) > 1
        ]
        return tokens
    except:
        return []


df = pd.read_csv('professional_financial_sentiment.csv')
df2 = pd.read_csv('casual_reddit_wallstreetbets_posts_text.csv')
dftest = pd.read_csv('reddit_financial_posts_complete_analysis.csv')
df3=pd.read_csv('reddit_financial_posts_classified_balanced.csv')
print(df3.columns)
df['tokenized_content'] = df['Sentence'].fillna('').apply(clean_and_tokenize)
df2['tokenized_content'] = df2['selftext'].fillna('').apply(clean_and_tokenize)
dftest['tokenized_content'] = dftest['tokenized_content'].fillna('').apply(clean_and_tokenize)

combined_data = df['tokenized_content'].tolist() + df2['tokenized_content'].tolist()
combined_data = [tokens for tokens in combined_data if len(tokens) > 0]

cbowmodel = Word2Vec(
    sentences=combined_data,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=0,
    epochs=10,
    alpha=0.025,
    min_alpha=0.0001
)

positive_words = [
    'profit', 'gain', 'win', 'success', 'profitable', 'winning', 'bull', 'bullish', 'up', 'rise', 'rising', 'growth',
    'growing', 'gains', 'winner', 'winners', 'green', 'positive', 'upward', 'climb', 'climbing', 'soar', 'soaring',
    'surge', 'boost', 'jump', 'rally', 'advance', 'strengthen', 'improve', 'increase', 'expand',
    'buy', 'long', 'hold', 'invest', 'investment', 'opportunity', 'potential', 'strong', 'solid', 'good', 'great',
    'excellent', 'accumulate', 'accumulating', 'undervalued', 'value', 'bargain', 'cheap', 'oversold', 'rebound',
    'recovery', 'turnaround', 'upside', 'conviction', 'confident', 'optimistic', 'promising', 'attractive',
    'favorable', 'beneficial', 'lucrative', 'rewarding', 'worthwhile',
    'breakout', 'breakthrough', 'high', 'peak', 'uptrend', 'momentum', 'strength', 'support', 'bounce', 'reversal',
    'squeeze', 'pump', 'blast', 'explode', 'expansion', 'acceleration', 'catalyst', 'trigger', 'upgrade', 'beat',
    'outperform', 'exceed', 'surprise', 'optimism',
    'revenue', 'earnings', 'eps', 'margin', 'dividend', 'yield', 'return', 'fundamentals', 'cash', 'liquid',
    'solvent', 'healthy', 'stable',
    'tendies', 'diamond', 'hands', 'hodl', 'stonks', 'moon', 'apes', 'gamma', 'drs', 'chad', 'based', 'legend',
    'king', 'alpha', 'epic', 'insane', 'crazy', 'wild', 'sick',
    'nice', 'cool', 'awesome', 'amazing', 'fantastic', 'wonderful', 'brilliant', 'smart', 'clever', 'perfect',
    'happy', 'excited', 'thrilled', 'love', 'like', 'enjoy', 'appreciate', 'impressed', 'surprised', 'wow'
]

highly_positive_words = [
    'rocket', 'lambo', 'millionaire', 'rich', 'wealthy', 'fortune', 'jackpot', 'goldmine', 'bonanza',
    'windfall', 'treasure', 'lottery', 'retirement', 'freedom', 'fire', 'mansion', 'yacht', 'ferrari',
    'incredible', 'outstanding', 'exceptional', 'phenomenal', 'spectacular', 'magnificent', 'extraordinary',
    'legendary', 'unbelievable', 'revolutionary', 'disruption', 'paradigm', 'historic', 'unprecedented',
    'record', 'breaking',
    'parabolic', 'exponential', 'vertical', 'nuclear', 'atomic', 'stratospheric', 'mind', 'blowing',
    'game', 'changer', 'life', 'changing', 'generational', 'wealth', 'billionaire',
    'yolo', 'ape', 'god', 'tier', 'galaxy', 'degen', 'sigma', 'emperor', 'deity', 'gigachad'
]

negative_words = [
    'loss', 'lose', 'losing', 'lost', 'bear', 'bearish', 'down', 'fall', 'falling', 'drop', 'dropping',
    'crash', 'dump', 'red', 'negative', 'decline', 'decrease', 'plunge', 'tank', 'crater', 'nosedive',
    'freefall', 'bloodbath', 'carnage', 'massacre', 'slaughter', 'destruction', 'obliteration',
    'annihilation', 'collapse', 'plummet',
    'recession', 'bubble', 'correction', 'weak', 'bottom', 'low', 'dip', 'panic', 'capitulation',
    'despair', 'fear', 'uncertainty', 'doubt', 'fud', 'resistance', 'rejection', 'breakdown', 'broken',
    'death', 'cross', 'flag', 'distribution', 'selling', 'pressure',
    'debt', 'bankruptcy', 'insolvency', 'default', 'writedown', 'impairment', 'provision', 'margin',
    'call', 'delisting', 'suspension', 'investigation', 'sec', 'lawsuit', 'fine', 'penalty', 'violation',
    'insolvent',
    'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'disgusting', 'stupid', 'dumb', 'foolish',
    'ridiculous', 'pathetic', 'useless', 'worthless', 'disappointed', 'frustrated', 'angry', 'mad',
    'upset', 'sad', 'depressed', 'worried', 'concerned', 'scared', 'afraid', 'nervous'
]



def calculate_sentiment_score(tokens):
    if not isinstance(tokens, list) or len(tokens) == 0:
        return 0, "neutral"

    pos_count = sum(1 for token in tokens if token in positive_words)
    highly_pos_count = sum(1 for token in tokens if token in highly_positive_words)
    neg_count = sum(1 for token in tokens if token in negative_words)

    total_sentiment_words = pos_count + highly_pos_count + neg_count
    total_words = len(tokens)

    positive_score = pos_count + (highly_pos_count * 2)
    negative_score = neg_count
    net_sentiment = positive_score - negative_score

    if total_words > 0:
        sentiment_density = total_sentiment_words / total_words
        length_factor = min(1.5, 1 + sentiment_density)
        net_sentiment *= length_factor

    return net_sentiment, {
        'pos_count': pos_count,
        'highly_pos_count': highly_pos_count,
        'neg_count': neg_count,
        'total_words': total_words,
        'sentiment_density': sentiment_density if total_words > 0 else 0
    }


def get_doc_vector(tokens, model):
    if not isinstance(tokens, list) or len(tokens) == 0:
        return np.zeros(model.wv.vector_size)

    vectors = [model.wv[word] for word in tokens if word in model.wv.key_to_index]
    if not vectors:
        return np.zeros(model.wv.vector_size)

    return np.mean(vectors, axis=0)


def get_sentiment_vectors(words, model):
    vectors = []
    for word in words:
        if word in model.wv.key_to_index:
            vectors.append(model.wv[word])

    if not vectors:
        return np.zeros(model.wv.vector_size)

    return np.mean(vectors, axis=0)


def classify_sentiment_balanced(tokens):
    if not isinstance(tokens, list) or len(tokens) == 0:
        return "Don't Recommend"

    sentiment_score, details = calculate_sentiment_score(tokens)
    doc_vector = get_doc_vector(tokens, cbowmodel)

    if not np.all(doc_vector == 0):
        pos_vector = get_sentiment_vectors(positive_words, cbowmodel)
        highly_pos_vector = get_sentiment_vectors(highly_positive_words, cbowmodel)
        neg_vector = get_sentiment_vectors(negative_words, cbowmodel)

        pos_sim = cosine_similarity([doc_vector], [pos_vector])[0][0] if np.any(pos_vector) else 0
        highly_pos_sim = cosine_similarity([doc_vector], [highly_pos_vector])[0][0] if np.any(highly_pos_vector) else 0
        neg_sim = cosine_similarity([doc_vector], [neg_vector])[0][0] if np.any(neg_vector) else 0

        w2v_score = (pos_sim + highly_pos_sim * 1.5) - neg_sim
        combined_score = sentiment_score * 0.7 + w2v_score * 0.3
    else:
        combined_score = sentiment_score

    if combined_score >= 2.0:
        return "Highly Recommend"
    elif combined_score >= 0.5:
        return "Recommend"
    elif combined_score <= -1.0:
        return "Don't Recommend"
    else:
        if details['neg_count'] > details['pos_count'] + details['highly_pos_count']:
            return "Don't Recommend"
        else:
            return "Recommend"


dftest['sentiment_label'] = dftest['tokenized_content'].apply(classify_sentiment_balanced)
dftest.to_csv('reddit_financial_posts_classified_balanced.csv', index=False)

print(f"Classified {len(dftest)} posts")
print(dftest['sentiment_label'].value_counts())