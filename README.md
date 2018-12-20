# CS-433 Project 2
## Damian Dudzicz, Guillaume Michel, Adrien Vandenbroucque

This repository contains the code for the Project 2 of EPFL's Machine Learning course. The goal of this project is to perform sentiment analysis over a dataset of Tweets.

## Packages required
In order to run the code properly, you will need the following packages (we provide the reader with the `pip install` command line for easy installation):

- NLTK (The Natural Language Processing Toolkit)
`sudo pip install -U nltk`
In this library, you will specifically need: 
	- PorterStemmer
	- TweetTokenizer
	- Corpus of stop-words
- SymSpellpy `pip install -U symspellpy` 
- NumPy `sudo pip install -U numpy`
- Scikit-Learn `pip install -U scikit-learn`
- Matplotlib `python -m pip install -U matplotlib`
- Gensim `pip install --upgrade gensim`

## How to run the program

You need to place first the full positive and negative Tweets Datasets `train_pos_full.txt` and `train_neg_full.txt` into the `/data` folder along the `test_data.txt` Tweets Test Dataset. The command `python3 run.py` will let you compute the predictions for the best model, and write the predictions for the test data into a file named `submission.csv`. 

## Structure of the files

### `helpers.py`
The file contains the function used to create the CSV submission for CrowdAI. 

### `preprocessing.py`
The file contains the functions to load and prepare the train and test data. It also contains the function used to tokenize the tweets. The steps such as stemming, spelling correction and diamond tags removal are implemented but not used in the final prediction creation.

### `w2v.py`
The file contains the function to create and train a w2v and fasttext model. It also contains the function used to convert tweets to vectors.

### `pipeline.py`
The file contains the function that creates a Scikit-Learn pipeline for using a Bag of Words representation of words with TF-IDF weighting implemented with the `TfidfVectorizer` class. To classify these vectors, we then use the LinearSVC classifier in the pipeline.

### `cross_validation.py`

The file contains the function that performs a repeated k-fold cross-validation (more precisely, stratified k-fold).

### `plot.py`
The file contains the function used to plot the results of the cross-validation.

### `cross_val_plotting.py`
The file contains the methods which generated the graphs presented in the report.

### `run.py`
The file contains the code that loads the training and test data. To be used in order to generate the best prediction.


__NOTE__: In order to exlusively generate the prediction with the `run.py` the `cross_validation.py`, `plot.py`,`plotting.py` and `w2v.py` are __not__ required.
