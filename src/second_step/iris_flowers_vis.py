''' We're using the iris dataset to train a binary tree classifier here,
and we'll be visualising the tree using scikit-learn'''

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
test_indices = [0, 50, 100]

#Split the dataset
#Training data
train_target = np.delete(iris.target, test_indices)
train_data = np.delete(iris.data, test_indices, axis = 0)

#Testing data
test_target = iris.target[test_indices]
test_data = iris.data[test_indices]

#Trainign our model
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

#Time to verify our tests
print(test_target)
print(clf.predict(test_data))


import graphviz
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")

dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph
