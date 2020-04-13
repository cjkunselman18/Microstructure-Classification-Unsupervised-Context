# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:18:06 2020

@author: Courtney
"""

from sklearn.semi_supervised import LabelPropagation

# This algorithm takes the label "-1" for the unlabeled data
label_prop_labels = label_list_train[:]
for i in range(0,36):
    label_prop_labels.append(-1)

    
label_prop = LabelPropagation('rbf', gamma=1,max_iter=100000)
label_prop.fit(no_test,label_prop_labels)
label_propagation_out = label_prop.transduction_
label_propagation_unknown_labels = label_propagation_out[48:84]
# Gamma = 10 or above caused numerical ill-conditioning; decided to go with 
# gamma = 1 to have smaller radius of influence (as gamma decreases, larger 
# classes in the training set tend to dominate)

