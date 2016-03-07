# Dependencies
# brew install graphviz

from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

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

    with open('tree.png', 'wb') as f:
        f.write(graph.create_png())

export_tree( clf, iris.feature_names )