from matplotlib import pyplot as plt

def plot_cross_validation(scores, stds):
    """Plot the accuracies computed during the
    k-fold cross-validation process"""

    plt.errorbar(x=list(range(len(scores))), y=scores, yerr=stds, marker='o', ecolor='orange')
    plt.xlabel('K-fold indice')
    plt.ylabel('Average accuracy')
    plt.title('Mean accuracies of the k-fold cross-validations')
    plt.show()
