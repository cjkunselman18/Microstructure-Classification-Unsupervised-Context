# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:50:28 2020

@author: Courtney
"""

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


# Kernel/Hyper-parameter selection
# Note that if there is a tie, linear kernels, and lower values of C and gamma 
# will be used

tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000,10000]},{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4,1e-2,1e-1],'C': [1,10,100, 1000,10000]}]

print("# Tuning hyper-parameters for accuracy")
print()

cv = KFold(n_splits = 5,random_state=2018)
clf = GridSearchCV(SVC(), tuned_parameters, cv=cv, scoring='accuracy' )
clf.fit(labeled_train, label_list_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))



baseline_svm = clf.best_estimator_
