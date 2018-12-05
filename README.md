# CS-433 Project 2
## Damian Dudzicz, Guillaume Michel, Adrien Vandenbroucque

This repository contains the code for the Project 2 of EPFL's Machine Learning course. The goal of this project is to perform sentiment analysis over a dataset of Tweets.

## Packages required

In order to run the code properly, you will need the following packages:
- NLTK (The Natural Language Processing Toolkit)
In this library, you wil specifically need to download: 
	- PorterStemmer
	- TweetTokenizer
	- Corpus of stop-words
- NumPy
- Scikit-Learn
- Matplotlib
- Gensim 

The structure of the files is the following:
- `helpers.py` contains the function useful for the CSV submission,
 - `preprocessing.py` contains functions useful to load and preprocess tweets,
- `w2v.py` contains functions useful for creating and training a Word2Vec model,
- `pipeline.py` contains the Scikit-Learn pipeline we used for the final version,
- `cross_validation.py` contains functions for performing  the cross validation method we used,
- `plot.py` contains functions to plot the results of the cross-validation,
- `Project2-ML.ipynb` is a notebook used when choosing which method was the best, and
- `run.py` contains the code to load the data and run one of the models in order to compute predictions.

The train and test data can be found in the `data/` folder. 

## How to run the program

The command `python3 run.py` will let you compute the predictions for the best model, and write the predictions for the test data into a file named `submission.csv`.

## More about the files

### `helpers.py`
This file contains the function used to create the CSV submission for CrowdAI. 

### `preprocessing.py`
This file contains the functions to load and prepare the train and test data. It also contains the function used to tokenize the tweets.

### `w2v.py`
This file contains the function to create and train a Word2Vec model, here we used a modified one called FastText. It also contains a function used to convert tweets to vectors.

### `pipeline.py`
This file contains the function that creates a Scikit-Learn pipeline for using a Bag of Words representation of words with TF-IDF weighting. To classify these vectors, we then use the LinearSVC classifier.

### `cross_validation.py`

This file contains the function that performs a repeated k-fold cross-validation (more precisely, stratified k-fold).

### `plot.py`
This file contains the function used to plot the results of the cross-validation.


### `Project2-ML.ipynb`
This notebook contains code that evaluate the different models, when trying to find the best.

### `run.py`
This file contains the code that loads the training and test data. 

We then compute the representation of tweets in a vector space and classify those. We use in this case the model that gave us the best results (TfIdfVectorizer + LinearSVC). We finfally compute and submit the predictions, in the format accepted by the CrowdAI competition.

