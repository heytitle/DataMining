# Draw Graph
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# Initial Color values
colors = ['red','blue','orange','cyan','black','pink']

def scatter(x, y, label_tex, marker,color):
	plt.scatter(x,y, marker = marker, color = color,label = label_tex)
def cost_line (slope, cross):
	# fpr X [0]
	# tpr Y [1]
	y_inter = cross[0] - slope * cross[1]
	plt.plot([0,1],[y_inter, y_inter + slope])


# # Calculate cost line properties
# cost_slope = 1.0/5.0
# cost_slope1 = 1.0
# y_intersect = tpr_b[1]-cost_slope*fpr_b[1]
# y_intersect1 =tpr_b[1] - cost_slope1 * fpr_b[1]

def show():
	plt.xlim((-0.2,1.2))
	plt.ylim((0,1.2))

	plt.legend(bbox_to_anchor=(1.05, 1), loc=5,fontsize = 8, scatterpoints =1)
	plt.title('ROC of A, B, and C with thresholds')

	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')

	plt.show()



	# for f,t, thres,color in zip(fpr_c,tpr_c,thresholds_c,colors):
	# plt.scatter(f,t,'>', color = color, label = 'C: ' + str(thres))
