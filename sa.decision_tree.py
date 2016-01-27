import math
import matplotlib.pyplot as pyplot
import numpy as np
import pydot 
import random
from sklearn import tree
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

ans_data = np.loadtxt('testing_data2.txt', delimiter='|')
ans_X = ans_data[:, :1000]

# For printing the decision trees themselves
dot_data = StringIO() 

# PART A

samples_list = []
training_loss_list = []
test_loss_list = []
training_loss_list_normalized = []
test_loss_list_normalized = []

lowest_loss = 1

for i in range(1,100):
    clf = tree.DecisionTreeClassifier()
    clf.min_samples_leaf = i
    samples_list.append(i)
    print i
    clf.fit(X,Y)
    # However, we likely also want it normalized.
    training_loss_list_normalized.append(1 - (clf.score(X,Y)))
    test_loss_list_normalized.append(1 - (clf.score(test_X, test_Y)))
    if test_loss_list_normalized[-1] < lowest_loss:
        lowest_loss = test_loss_list_normalized[-1]
        best_clf = clf

training_loss_list_normalized.append("B")
test_loss_list_normalized.append("B")
samples_list.append("B")

# PART B
# Now we want to do the same thing, but instead modifying the maximal
# tree depth. 

for i in range(2,100):
    print i
    clf = tree.DecisionTreeClassifier()
    clf.max_depth = i
    samples_list.append(i)
    clf.fit(X,Y)

    # However, we likely also want it normalized.
    training_loss_list_normalized.append(1 - (clf.score(X,Y)))
    test_loss_list_normalized.append(1 - (clf.score(test_X, test_Y)))
    if test_loss_list_normalized[-1] < lowest_loss:
        lowest_loss = test_loss_list_normalized[-1]
        best_clf = clf

# Once we are done, we want to put the optimal thing in a solution set.
# We also want the name to be unique
ans_Y = best_clf.predict(ans_X)
f = open("solution_{}_{}".format(lowest_loss, str(datetime.datetime.now())), 'w')
f.write("Id,Prediction\n")
for i,j in enumerate(ans_Y):
    f.write("{},{}\n".format(i+1,int(j)))
#we really like number 56 from the depth structure idk why
