'''We're finally getting to write our own verion of the K Nearest Neighbors classifier'''
from scipy.spatial import distance

def euc(a, b):
    return distance.euclidean(a, b)

class scrappyKNN():
    #Our fit training method
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    #Our prediction method
    def predict(self, x_test):
        predictions = []

        for i in x_test:
            label = self.closest(i)
            predictions.append(label)

        return predictions

    def closest(self, row):
        best_dist = euc(row, self.x_train[0])
        best_index = 0

        for i in range(1, len(self.x_train)):
            dist = euc(row, self.x_train[i])
            
            if dist < best_dist:
                best_dist = dist
                best_index = i

        return self.y_train[best_index]

from sklearn import datasets
iris = datasets.load_iris()

x = iris.data   #x is our data
y = iris.target #y is out labels

from sklearn.cross_validation import train_test_split

#Time to split our data into testing and training sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

#Now tryout our own KNN Classifier
#from sklearn.neighbors import KNeighborsClassifier
clf = scrappyKNN()

clf.fit(x_train, y_train)

prediction = clf.predict(x_test)

#Get out new classifiers accuracy score
from sklearn.metrics import accuracy_score
print("\n\naccuracy using the k neighbors classifier is", accuracy_score(y_test, prediction))

