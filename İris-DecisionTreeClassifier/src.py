# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 16:29:57 2020

@author: alida
"""

#import necessery libraries

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()

x = iris.data
y = iris.target
y_names = iris.target_names

test_ids = np.random.permutation(len(x))

#manually split data to train and test sublists, test wil be the last 10 entries
x_train = x[test_ids[:-10]]
x_test = x[test_ids[-10:]]
y_train= y[test_ids[:-10]]
y_test = y[test_ids[-10:]]


#creating a model as DecisionTreeClassifier

clf = DecisionTreeClassifier()

#training (fitting) the train data with model
clf.fit(x_train,y_train)

# after training, we are predicting y_Test values with x_test values 
predict = clf.predict(x_test)

# after the prediction is done, lets calculate the accuracy score 

accuarcy = accuracy_score(predict, y_test)*100

#now lets print our predicted values, test values and the accuracy score

print("actual values = {act} ".format(act = y_test))
print("predicted values =  {pred}".format(pred=predict))
print("the accuracy score is {}".format(accuarcy))

#actual values = [2 2 2 1 0 1 0 0 1 2] 
#predicted values =  [2 2 2 1 0 1 0 0 1 2]
#the accuracy score is 100.0