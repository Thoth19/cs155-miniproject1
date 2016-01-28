import math
import matplotlib.pyplot as pyplot
import numpy as np
import pydot 
import random
from sklearn import tree
from sklearn import cross_validation
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals.six import StringIO  
import datetime

# Load Data
# Note that I removed the top line so we could pull the data 
# with np.loadtxt

data = np.loadtxt('training_data2.txt', delimiter='|')
X = data[0:3351, :1000]
Y = data[0:3351, 1000]

test_X = data[3351:, :1000]
test_Y = data[3351:, 1000]
X, test_X, Y, test_Y = cross_validation.train_test_split(data[:, :1000], data[:,1000],test_size=0.4)

ans_data = np.loadtxt('testing_data2.txt', delimiter='|')
ans_X = ans_data[:, :1000]

clf = AdaBoostClassifier(n_estimators=10000, base_estimator=tree.DecisionTreeClassifier(compute_importances=None, criterion='gini',
            max_depth=10, max_features=None, min_density=None,
            min_samples_leaf=1, min_samples_split=2, random_state=None,
            splitter='best'))
clf.fit(X,Y)

loss = 1 - clf.score(test_X, test_Y)


# Once we are done, we want to put the thing in a solution set.
# We also want the name to be unique
ans_Y = clf.predict(ans_X)
f = open("solution_{}_{}".format(loss, str(datetime.datetime.now())), 'w')
f.write("Id,Prediction\n")
for i,j in enumerate(ans_Y):
    f.write("{},{}\n".format(i+1,int(j)))
#we really like number 56 from the depth structure idk why
