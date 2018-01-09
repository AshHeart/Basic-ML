'''Here we'll be exploring the iris data set which we'll be using in the next file
This is just to see how to access the various features and targets in the dataset'''

from sklearn.datasets import load_iris

iris = load_iris()

print("Our feature names are: ", iris.feature_names)
print("Our target names are: ", iris.target_names)

'''we can access the actual data using data'''
print("The first flower has: ", iris.data[0])
print("And our first flower is: ", iris.target[0], "which is a ", iris.target_names[0])

'''let's see th eentire dataset now'''
for i in range(len(iris.target)):
    print('{}. Label: {}, Features: {}'.format(i, iris.target[i], iris.data[i]))
