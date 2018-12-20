import numpy as np
import string
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


def load_train_tweets(name_pos, name_neg):
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
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet = tweet.replace('[^\w\s]','')
    tokens = tokenizer.tokenize(tweet)

    # Option to perform stemming
    if stem:
        tokens = [stemmer.stem(token) for token in tokens]

    # Option to remove stop-words
    if remove_stop_words:
        tokens = [token for token in tokens if token not in stop]

    return tokens

def correct_spelling(sentences, max_edit_distance_lookup=2):
    """Correct the spelling of the tweets to have a more accurate analysis"""
    initial_capacity = 83000
    max_edit_distance_dictionary = 2
    prefix_length = 7
    sym_spell = SymSpell(initial_capacity, max_edit_distance_dictionary,prefix_length)

    dictionary_path = os.path.join(os.path.dirname(__file__), "./data/frequency_dictionary_en_82_765.txt")
    term_index = 0  # column of the term in the dictionary text file
    count_index = 1  # column of the term frequency in the dictionary text file
    if not sym_spell.load_dictionary(dictionary_path, term_index, count_index):
        print("Dictionary file not found")
        return

    corrected_sentences = []
    # take best suggestion of correction
    for input in sentences:
        suggestions = sym_spell.lookup_compound(input, max_edit_distance_lookup)
        corrected_sentences.append(suggestions[0].term)

    return corrected_sentences

def removeDiamonds(sentences):
    """Remove the diamonds tags (i.e < url > and < user >) from the tweets"""

    # creating the set of sentences without diamonds
    newSentences = []
    for sentence in sentences:
        inDiamond=False
        currentIndex=0
        newSentence=""
        for i in range(len(sentence)):
            c = sentence[i]
            if c == '<':
                newSentence += sentence[currentIndex:i]
                inDiamond = True
                currentIndex = i
            if c == '>' and inDiamond:
                currentIndex = i+1
                inDiamond = False
        newSentence += sentence[currentIndex:len(sentence)]
        newSentences.append(newSentence)
    return newSentences