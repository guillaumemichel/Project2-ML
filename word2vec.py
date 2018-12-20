import numpy as np
from gensim.models.doc2vec import TaggedDocument
from gensim.models import FastText
from gensim.models import Word2Vec


def create_w2v(xtrain, dim, iter, fasttext=True):
    """Create and train a FastText model using training tweets"""

    # Convert data to Tagged document in order to work with w2v
    tagged_data = [TaggedDocument(words=tweet, tags=[str(i)]) for i,tweet in enumerate(xtrain)]

    #Get the words
    words = [t.words for t in tagged_data]

    # Create and train a w2v model
    if fasttext = True:
        model = FastText(size=dim, workers=4, iter=iter, word_ngrams=1)
    else:
        model = Word2Vec(size=dim, workers=4, iter=iter)
        
    model.build_vocab(words)
    model.train(words, total_examples=fast.corpus_count, epochs=fast.epochs)

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
