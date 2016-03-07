import util
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix;

X, y, attribute_names = util.load(10)
df = pd.DataFrame(X, columns=attribute_names)

df['class'] = y
print "There are %d instances with %d columns" %( len(df), len(df.columns) )

mask = np.in1d( df['class'], [1,2] );
df = df[ mask ]
print "Filter out minority classes ( %d left )" % ( len(df) )

features = df.columns;
classes =["Metastates","Malign Lymph"]

seeds = [
    10,20,30
];

for s in seeds:
    print "Seed : %d" %( s );
    train, test = train_test_split(df, test_size = 0.2, random_state=s)
    print "Training : %d ( %.2f )" % ( len(train), len(train)*1.0/len(df) )
    print "Testing : %d ( %.2f )" % ( len(test), len(test)*1.0/len(df) )

    clf = tree.ExtraTreeClassifier();
    clf = clf.fit( train[features[:-1]], train['class'] )
    pred_test = clf.predict( test[features[:-1]] );
    print "Accuracy %.4f" % ( accuracy_score(test['class'],pred_test) )

    util.export_tree( clf, features, classes, filename="random-tree-"+str(s)+".png" )
