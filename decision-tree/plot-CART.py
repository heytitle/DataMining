# Dependencies
# brew install graphviz
# coding=utf-8

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

from sklearn.datasets import load_iris
from sklearn import tree
from numpy import array
import pandas as pd
import random
from sklearn.cross_validation import train_test_split

from load_data import load

X, y, attribute_names = load(10)
# X, y, attribute_names = dataset.get_dataset(target=dataset.default_target_attribute, return_attribute_names=True)
lymph = pd.DataFrame(X, columns=attribute_names)
lymph['class'] = y
# remove class 0 and class 3
lymph_filtered = lymph[(lymph['class']==1) | (lymph['class']==2)]
# Split to validate
X_train, X_test, y_train, y_test = train_test_split(lymph_filtered[attribute_names], lymph_filtered['class'], test_size=0.20, random_state=42)

# Draw decision tree
clf = tree.DecisionTreeClassifier()
lymph_tree = clf.fit(X_train, y_train)
export_tree(lymph_tree,attribute_names)



from sklearn.externals.six import StringIO
from IPython.display import Image
import pydot

def export_tree( clf, feature_names ):

    dot_data = StringIO()

    tree.export_graphviz(clf, out_file=dot_data,
                            feature_names=feature_names,
                            filled=True, rounded=True,
                            special_characters=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())

    with open('lymph_tree.png', 'wb') as f:
        f.write(graph.create_png())

