''' Install scikit learn by running sudo pip install -U scikit-learn '''

from sklearn import tree

'''Our datasets'''
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]

'''Classifier we use, a binary or tree classifier'''
clf = tree.DecisionTreeClassifier()

'''Learning algorithm fit'''
clf = clf.fit(features, labels)

'''Test out model'''
res = clf.predict([[160, 0]])

if res == 1:
    print("Orange")
else: 
    print("Apple")
