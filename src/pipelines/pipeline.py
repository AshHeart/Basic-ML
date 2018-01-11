'''Aw yeah time to actually get to the heart of machine learning, 
here we learn to split training data and testing data and try a
few classfifiers and comapre their results'''

from sklearn import datasets
iris = datasets.load_iris()

x = iris.data   #x is our data
y = iris.target #y is out labels

from sklearn.cross_validation import train_test_split

#Time to split our data into testing and training sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

#Train our model first using our old DecisionTree Classifier
from sklearn import tree
clf = tree.DecisionTreeClassifier()

clf.fit(x_train, y_train)

prediction = clf.predict(x_test)

#Get out classifiers accuracy score
from sklearn.metrics import accuracy_score
print("\n\n\n\naccuracy using our tree classifier is", accuracy_score(y_test, prediction))


#Now tryout the new guy the KNeighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()

clf.fit(x_train, y_train)

prediction = clf.predict(x_test)

#Get out new classifiers accuracy score
from sklearn.metrics import accuracy_score
print("\n\naccuracy using the k neighbors classifier is", accuracy_score(y_test, prediction))

