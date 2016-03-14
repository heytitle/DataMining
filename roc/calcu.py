import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


# # Lists that save values
# predict = []
# fpr = []
# tpr = []
# auc = []

def check_threshold(value, threshold):
	predict = []
	for i in range(0,len(value)):
		predict.append(value[i]>= threshold)
	return predict

def roc(predict,true):
	f, t, thresholds = roc_curve(true, predict)
	# roc_auc = auc(fpr,tpr)
	auc = 1
	return f, t, auc, thresholds



