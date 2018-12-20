import numpy as np
from gensim.models.doc2vec import TaggedDocument
from gensim.models import FastText
from gensim.models import Word2Vec


def create_fast(xtrain, dim, iter):
    """Create and train a FastText model using training tweets"""

    # Convert data to Tagged document in order to work with w2v
    tagged_data = [TaggedDocument(words=tweet, tags=[str(i)]) for i,tweet in enumerate(xtrain)]

    #Get the words
    words = [t.words for t in tagged_data]

    # Create and train a FastText model
    fast = FastText(size=dim, workers=4, iter=iter, word_ngrams=1)
    fast.build_vocab(words)
    fast.train(words, total_examples=fast.corpus_count, epochs=fast.epochs)

    return fast

def create_w2v(xtrain,dim):
    model = Word2Vec(xtrain, size=dim, window=5, min_count=1, workers=4)
    return model
    
def tweets2vec(tweets, dim, model):
    """Convert a list of tweets to their vector representation
    by averaging the vector representation of the words in the
    tweets"""

    def tweet2vec(tweet, model):
        """Convert a tweet to its vector representation
        by averaging the vector representation of the words in the
        tweet"""

        # Get the number of tokens in the tweet
        nwords = len(tweet)

        # If there are more than one word, compute average
        if nwords > 0:
            vector = np.zeros(dim)
            vector = np.mean([model[w] for w in tweet if w in model], axis=0)
        else:# Else return the zero vector
            vector = np.zeros(dim)

        return vector

    return [tweet2vec(tweet, model) for tweet in tweets]
