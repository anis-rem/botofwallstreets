import pandas as pd
from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import ast  # For safely evaluating string representations of lists

# Load the dataset
df = pd.read_csv('reddit_financial_posts.csv')
print("Dataset shape:", df.shape)
print(df.head())

# Check the format of tokenized_content
print("\nSample of tokenized_content:")
print(df['tokenized_content'].iloc[0])
print("Type:", type(df['tokenized_content'].iloc[0]))
data = df['tokenized_content'].dropna()
if isinstance(data.iloc[0], str):
    try:
        data = data.apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
        print("Successfully converted string representations to lists")
    except (ValueError, SyntaxError):
        data = data.apply(lambda x: x.split() if pd.notna(x) else [])
        print("Split strings into token lists")
data = [tokens for tokens in data if len(tokens) > 0]

print(f"\nNumber of documents: {len(data)}")
print(f"Sample tokenized document: {data[0][:10]}...")
cbowmodel = Word2Vec(
    sentences=data,
    min_count=1,        # Minimum word frequency
    vector_size=100,    # Dimensionality of word vectors
    window=5,           # Context window size
    sg=0,              # Use CBOW (sg=1 for Skip-gram)
    workers=4,         # Number of worker threads
    epochs=10,         # Number of training epochs
    alpha=0.025,       # Initial learning rate
    min_alpha=0.0001   # Minimum learning rate
)

print(f"\nModel trained successfully!")
print(f"Vocabulary size: {len(cbowmodel.wv.key_to_index)}")
print(f"Vector size: {cbowmodel.wv.vector_size}")
sample_words = ['money', 'stock', 'investment', 'market', 'bank']
available_words = [word for word in sample_words if word in cbowmodel.wv.key_to_index]

if available_words:
    print(f"\nTesting with available words: {available_words}")
    for word in available_words[:3]:
        try:
            similar_words = cbowmodel.wv.most_similar(word, topn=5)
            print(f"Words similar to '{word}': {similar_words}")
        except KeyError:
            print(f"Word '{word}' not in vocabulary")
else:
    vocab_sample = list(cbowmodel.wv.key_to_index.keys())[:10]
    print(f"Sample vocabulary words: {vocab_sample}")
cbowmodel.save("reddit_financial_word2vec.model")
print("\nModel saved as 'reddit_financial_word2vec.model'")
cbowmodel.wv.save("reddit_financial_word_vectors.kv")
print("Word vectors saved as 'reddit_financial_word_vectors.kv'")