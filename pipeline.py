import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import re

def pipeline_model():
    return Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=preprocessing.tokenize, ngram_range=(1, 3))),
        ('classifier', LinearSVC()),
        ])
