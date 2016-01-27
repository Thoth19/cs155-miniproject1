import math
import matplotlib.pyplot as pyplot
import numpy as np
import pydot 
import random
from sklearn import tree
from sklearn.externals.six import StringIO  

# Load Data
# Note that I removed the top line so we could pull the data 
# with np.loadtxt

data = np.loadtxt('training_data2.txt', delimiter='|')
X = data[0:3351, :1000]
Y = data[0:3351, 1000]

test_X = data[3351:, :1000]
test_Y = data[3351:, 1000]

# For printing the decision trees themselves
dot_data = StringIO() 

# PART A

samples_list = []
training_loss_list = []
test_loss_list = []
training_loss_list_normalized = []
test_loss_list_normalized = []
for i in range(1,26):
    clf = tree.DecisionTreeClassifier()
    clf.min_samples_leaf = i
    samples_list.append(i)
    print i
    clf.fit(X,Y)
    # We can find the classifcation error using the score
    # function (which is accuracy) and multiply by the sample size
    training_loss_list.append(400*(1 - clf.score(X,Y)))
    test_loss_list.append(169*(1 - clf.score(test_X, test_Y)))
    # However, we likely also want it normalized.
    training_loss_list_normalized.append(1 - (clf.score(X,Y)))
    test_loss_list_normalized.append(1 - (clf.score(test_X, test_Y)))

    # If we want to print out the tree as a diagram
    # tree.export_graphviz(clf, out_file=dot_data) 
    # graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
    # graph.write_pdf("training{}.pdf".format(i))

# fig, ax = pyplot.subplots()
# l1, = pyplot.plot(samples_list, training_loss_list, label="Training")
# l2, = pyplot.plot(samples_list, test_loss_list, label="Testing")
# legend = ax.legend(loc='upper center', shadow=True)
# pyplot.title('Leaf Node Size vs Classification Error')
# pyplot.show()
fig, ax = pyplot.subplots()
l1 = pyplot.plot(samples_list, training_loss_list_normalized, label="Training")
l2 = pyplot.plot(samples_list, test_loss_list_normalized, label="Testing")
legend = ax.legend(loc='upper center', shadow=True)
pyplot.title('Leaf Node Size vs Classification Error (Normalized)')
pyplot.show()

# PART B
# Now we want to do the same thing, but instead modifying the maximal
# tree depth. 

depth_list = []
training_loss_list = []
test_loss_list = []
training_loss_list_normalized = []
test_loss_list_normalized = []
for i in range(2,21):
    print i
    clf = tree.DecisionTreeClassifier()
    clf.max_depth = i
    depth_list.append(i)
    clf.fit(X,Y)
    # We can find the classifcation error using the score
    # function (which is accuracy) and multiply by the sample size
    training_loss_list.append(400*(1 - clf.score(X,Y)))
    test_loss_list.append(169*(1 - clf.score(test_X, test_Y)))
    # However, we likely also want it normalized.
    training_loss_list_normalized.append(1 - (clf.score(X,Y)))
    test_loss_list_normalized.append(1 - (clf.score(test_X, test_Y)))

    # If we want to print out the tree as a diagram
    # tree.export_graphviz(clf, out_file=dot_data) 
    # graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
    # graph.write_pdf("training{}.pdf".format(i))

# fig, ax = pyplot.subplots()
# l1, = pyplot.plot(depth_list, training_loss_list, label="Training")
# l2, = pyplot.plot(depth_list, test_loss_list, label="Testing")
# legend = ax.legend(loc='upper center', shadow=True)
# pyplot.title('Maximum Depth vs Classification Error')
# pyplot.show()
fig, ax = pyplot.subplots()
l1 = pyplot.plot(depth_list, training_loss_list_normalized, label="Training")
l2 = pyplot.plot(depth_list, test_loss_list_normalized, label="Testing")
legend = ax.legend(loc='upper center', shadow=True)
pyplot.title('Maximum Depth vs Classification Error (Normalized)')
pyplot.show()