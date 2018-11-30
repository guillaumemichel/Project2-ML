import numpy as np
import string
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


def load_tweets(name_pos, name_neg):
    """Load the positive and negative tweets for training self.
    Returns tweets, and number of tweets of each sentiment
    (useful to set prediction values)"""

    # List to store the tweets
    sentences = []

    # Load the positive tweets
    with open(name_pos) as pos_f:
        for line in pos_f:
            sentences.append(line)

    # Number of positive tweets
    size_pos = len(sentences)

    # Load the negative tweets
    with open(name_neg) as neg_f:
        for line in neg_f:
            sentences.append(line)

    # Number of negative tweets
    size_neg = len(sentences) - size_pos

    return sentences, size_pos, size_neg

def load_test_tweets(name):
    """Load the test tweets and take care of the indices"""

    # Lists to store the tweets and their ids
    sentences = []
    ids = []

    # Load the tweets and ids
    with open(name) as f:
        for line in f:
            # Split on the first "," to get the id and tweet
            id, s = line.split(',', 1)
            
            sentences.append(s)
            ids.append(id)

    return sentences, ids

def predictions(size_pos, size_neg):
    """Return the list of labels for training tweets"""

    # Compute pred using the number of positive and negative tweets
    pred = np.concatenate([np.ones(size_pos), np.full(size_neg, -1)])

    return pred

def tokenize(tweet, stem=False, remove_stop_words=False):
    """Tokenize a given tweet"""

    stop = stopwords.words('english')
    stemmer = PorterStemmer()
    tokenizer = TweetTokenizer()
    tweet = tweet.replace('[^\w\s]','')
    tokens = tokenizer.tokenize(tweet)

    if stem:
        tokens = [stemmer.stem(token) for token in tokens]

    if remove_stop_words:
        tokens = [token for token in tokens if token not in stopwords]

    return tokens
