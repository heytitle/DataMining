import util
import pandas as pd
import numpy as np

from operator import itemgetter

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from time import time

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.grid_search import RandomizedSearchCV
from sklearn import tree

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

train, test = train_test_split(df, test_size = 0.2, random_state=20, stratify = df['class'] )


clf = tree.DecisionTreeClassifier()

n_iter_search = 20


param_dist = {
    "criterion": ['gini', 'entropy'],
    "min_samples_leaf": np.arange(5,20),
    "max_depth": np.arange(5,10)
}

random_search = RandomizedSearchCV(
    clf,
    param_distributions=param_dist,
    n_iter=n_iter_search,
    scoring = "accuracy"
)

start = time()

kf = StratifiedKFold( train['class'], n_folds=5 )

res = []
for train_index, test_index in kf:
    inner_train = train.iloc[train_index]
    inner_test = train.iloc[test_index]
    random_search.fit(inner_train[attribute_names[:-1]], inner_train['class'])

    pred_test = random_search.best_estimator_.predict( inner_test[attribute_names[:-1]] )
    accu = accuracy_score( inner_test['class'], pred_test )
    d = dict(
        clf = random_search.best_estimator_,
        score = accu,
        params = random_search.best_params_
    )
    res.append(d)

res = sorted( res, reverse = True, key=lambda i: i['score'] )

for i, d in enumerate(res):
    print("Model with rank: {0}".format(i + 1))
    print("Score: {0:.3f}".format( d['score'] ) )
    print("Parameters: {0}".format(d['params']))


best = res[0]['clf']
pred_test = best.predict( test[attribute_names[:-1]] )
print "-----------"
print "Accuracy %.4f" % ( accuracy_score(test['class'],pred_test) )
print classification_report( test['class'],pred_test )
