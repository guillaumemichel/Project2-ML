import preprocessing, word2vec, pipeline, helpers, cross_validation, plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt

print('Loading the dataset...')
tweets, size_pos, size_neg = preprocessing.load_tweets("./data/train_pos.txt", "./data/train_neg.txt")

pred = preprocessing.predictions(size_pos, size_neg)

print('Tokenizing...')
tokenized_tweets = [preprocessing.tokenize(tweet) for tweet in tweets]

#x_train, x_test, y_train, y_test = train_test_split(tokenized_tweets, pred, test_size = 0.2)
#x_test, ids_test = preprocessing.load_test_tweets('./test_data.txt')

print('Training the W2V model...')
model = word2vec.create_train_w2v(tokenized_tweets, 250, 30)

print('Converting tweets to vectors...')
x_train_vec = word2vec.tweets2vec(tokenized_tweets, 250, model)
#x_test_vec = word2vec.tweets2vec(x_test, 250, model)

print('Training classifier...')
clf = LogisticRegression(solver='lbfgs')
"""clf.fit(x_train_vec, y_train)
print('Accuracy: ', clf.score(x_test_vec, y_test))

clf = pipeline.pipeline_model()"""

cross_val_scores, stds = cross_validation.n_kfold_cross_validation(x_train_vec, pred, clf, n=5, k=5)

clf1 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(64, 4))
cross_val_scores1, stds1 = cross_validation.n_kfold_cross_validation(x_train_vec, pred, clf1, n=5, k=5)

clf2 = LinearSVC()
cross_val_scores2, stds2 = cross_validation.n_kfold_cross_validation(x_train_vec, pred, clf2, n=5, k=5)

plt.errorbar(x=list(range(len(cross_val_scores))), y=cross_val_scores, yerr=stds, marker='o', ecolor='orange', label='LogisticRegression')
plt.errorbar(x=list(range(len(cross_val_scores1))), y=cross_val_scores1, yerr=stds1, linestyle='--', marker='o', ecolor='orange', label='MLPClassifier')
plt.errorbar(x=list(range(len(cross_val_scores2))), y=cross_val_scores2, yerr=stds2, linestyle=':', marker='o', ecolor='orange', label='LinearSVC')
plt.xlabel('K-fold indice')
plt.ylabel('Average accuracy')
plt.title('Mean accuracies of the k-fold cross-validations')
plt.legend()
plt.show()
#plot.plot_cross_validation(cross_val_scores, stds)

cross_validation.display_info(cross_val_scores)
cross_validation.display_info(cross_val_scores1)
cross_validation.display_info(cross_val_scores2)

#helpers.create_csv_submission(ids_test, clf.predict(x_test), "submission.csv")
