import util
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;
import collections;

from operator import itemgetter

from sklearn.cross_validation import train_test_split
from sklearn import svm
from time import time

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.grid_search import RandomizedSearchCV


def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

X, y, attribute_names = util.load(32);

df = pd.DataFrame(X, columns=attribute_names)
df['class'] = y
print "There are %d instances with %d columns" %( len(df), len(df.columns) )
print df['class']

train, test = train_test_split(df, test_size = 0.2, random_state=20, stratify = df['class'] )


clf = svm.LinearSVC();
# clf = svm.LinearSVC(loss="hinge", multi_class = "crammer_singer" )

n_iter_search = 20


param_dist = {
    "loss": [ "hinge", "squared_hinge" ],
    "multi_class": ["ovr", "crammer_singer" ],
    "C": np.logspace(-4.5, -2, 10)
}

random_search = RandomizedSearchCV(
    clf,
    param_distributions=param_dist,
    n_iter=n_iter_search,
    scoring = "accuracy"
)

start = time()
random_search.fit(train[attribute_names[:-1]], train['class'])
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)

# clf = clf.fit( train[attribute_names[:-1]], train['class'] )

# pred_test = clf.predict( test[attribute_names[:-1]] )

# print "Accuracy %.4f" % ( accuracy_score(test['class'],pred_test) )
# print classification_report( test['class'],pred_test )
