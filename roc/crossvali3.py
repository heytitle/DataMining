from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, cross_validation, datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets

import graph as ga

# Choices of K
k = np.array(range(1,50))
# Array to store accuracy values
accu = []
# Random sample
n_samples = 1000
X, y = make_blobs(n_samples=n_samples, centers=3, n_features=2, random_state=2)
# Accuracy array
for i in k:
    # Initialize Classifier
    clf = KNeighborsClassifier(n_neighbors= i)
    a = cross_validation.cross_val_score(clf,X,y,scoring = 'accuracy', cv = 10)
    ave_a = np.mean(a)
    accu.append(ave_a)
mce_accu = 1- np.array(accu)
# Plot misclassification Rate
ga.plot_accu(k,accu,None)

precision_score(y_true, y_pred, average='micro') 








