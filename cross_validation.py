import numpy as np
import pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

def n_kfold_cross_validation(x, y, clf, n, k):
    """Perform k-fold cross-validation n times, and
    returns the obtained accuracies"""

    # List that will contain the score of each k-fold
    means = []
    stds = []

    # Perform k-fold cross-val n times
    for _ in range(n):
        skf = StratifiedKFold(n_splits=k, shuffle=True)
        # Compute the mean of scores of the k-fold
        list_scores = cross_val_score(clf, x, y, cv=skf, n_jobs=-1)
        print(list_scores)
        # Compute mean and standard deviation
        mean_score = np.mean(list_scores)
        std_score = np.std(list_scores)

        # Store them
        means.append(mean_score)
        stds.append(std_score)

    return means, stds

def display_info(scores):
    print('Mean accuracy is :', np.mean(scores))
    print('Standard deviation is: ', np.std(scores))
