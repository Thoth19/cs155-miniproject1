import math
import matplotlib.pyplot as pyplot
import numpy as np
import pydot 
import random
from sklearn import tree
from sklearn import cross_validation
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.six import StringIO  
import datetime


def print_to_file(clf):
    # Once we are done, we want to put the thing in a solution set.
    # We also want the name to be unique
    ans_Y = clf.predict(ans_X)
    with open("solution_score{}_{}".format(scores[clf], str(datetime.datetime.now())), 'w') as f:
        f.write("Id,Prediction\n")
        for i,j in enumerate(ans_Y):
            f.write("{},{}\n".format(i+1,int(j)))    


# Load Data
# Note that I removed the top line so we could pull the data 
# with np.loadtxt

data = np.loadtxt('training_data2.txt', delimiter='|')
X, test_X, Y, test_Y = cross_validation.train_test_split(data[:, :1000], data[:,1000],test_size=0.4)
temp1 = data[:, :1000]
temp2 = data[:, 1000]

ans_data = np.loadtxt('testing_data2.txt', delimiter='|')
ans_X = ans_data[:, :1000]

scores = {}
clfs = []
for i in range(20,75):
    clfs.append(AdaBoostClassifier(n_estimators=i, base_estimator=tree.DecisionTreeClassifier(compute_importances=None, criterion='gini',
            max_depth=1, max_features=None, min_density=None,
            min_samples_leaf=1, min_samples_split=2, random_state=None,
            splitter='best')))
    clfs[-1].fit(temp1,temp2)
    scores[clfs[-1]] = min(cross_validation.cross_val_score(clfs[-1], temp1, temp2, n_jobs=-1))
    print_to_file(clfs[-1])

for i in range(1,11):
    for j in range(1,11):
        clfs.append(RandomForestClassifier(max_depth=i, min_samples_leaf=j, n_jobs=-1))
        clfs[-1].fit(temp1, temp2)
        scores[clfs[-1]] = min(cross_validation.cross_val_score(clfs[-1], temp1, temp2, n_jobs=-1))
        print_to_file(clfs[-1])

best_score = 0
for i in scores:
    if scores[i] > best_score:
        best_score = scores[i]
        best_clf = i
ans_Y = i.predict(ans_X)
with open("solutionSpecial_score{}_{}".format(scores[best_clf], str(datetime.datetime.now())), 'w') as f:
    f.write("Id,Prediction\n")
    for i,j in enumerate(ans_Y):
        f.write("{},{}\n".format(i+1,int(j)))