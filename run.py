import preprocessing, pipeline, helpers

print('Loading the dataset...')
tweets, size_pos, size_neg = preprocessing.load_train_tweets("./data/train_pos_full.txt", "./data/train_neg_full.txt")

# creating the predictions
pred = preprocessing.predictions(size_pos, size_neg)

# loading the test tweets
x_test, ids_test = preprocessing.load_test_tweets('./data/test_data.txt')

print('Training classifier...')
#Best pipeline with TF-IDFVectorizer with gram = (1,4) and LinearSVC which yielded the best results on crowdai.org
clf = pipeline.pipeline_model(1,4)
clf.fit(tweets, pred)

# create the submission csv file
helpers.create_csv_submission(ids_test, clf.predict(x_test), "submission.csv")
