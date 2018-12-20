import preprocessing, word2vec, pipeline, helpers, cross_validation, plot
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt

print('Loading the dataset...')
tweets, size_pos, size_neg = preprocessing.load_train_tweets("./data/train_pos.txt", "./data/train_neg.txt")
pred = preprocessing.predictions(size_pos, size_neg)

print('Tokenizing...')
tokenized_tweets = [preprocessing.tokenize(tweet) for tweet in tweets]

print('Training the fasttext model...')
model = word2vec.create_w2v(tokenized_tweets, dim = 250, iter = 30)

print('Converting tweets to vectors...')
x_train_vec = word2vec.tweets2vec(tokenized_tweets, dim = 250, model = model)

print('Training classifier...')
clf = LogisticRegression(solver='lbfgs')
cross_val_scores, stds = cross_validation.n_kfold_cross_validation(x_train_vec, pred, clf, n=5, k=5)

clf1 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(64, 4))
cross_val_scores1, stds1 = cross_validation.n_kfold_cross_validation(x_train_vec, pred, clf1, n=5, k=5)

clf2 = LinearSVC()
cross_val_scores2, stds2 = cross_validation.n_kfold_cross_validation(x_train_vec, pred, clf2, n=5, k=5)

plt.plot(list(range(len(cross_val_scores1))), cross_val_scores1, marker='o', label='LogisticRegression')
plt.plot(list(range(len(cross_val_scores2))), cross_val_scores2, marker='o', label='MLPClassifier')
plt.plot(list(range(len(cross_val_scores3))), cross_val_scores3, marker='o', label='LinearSVC')
plt.xlabel('K-fold indice')
plt.ylabel('Average accuracy')
plt.title('Mean accuracies of the k-fold cross-validations')
plt.legend()

cross_validation.display_info(cross_val_scores)
cross_validation.display_info(cross_val_scores1)
cross_validation.display_info(cross_val_scores2)
