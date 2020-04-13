# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:11:11 2020

@author: Courtney
"""

# find labels for first iteration
unknown_labels = baseline_svm.predict(ambig_feat)
label_list_retrain = label_list_train + list(unknown_labels)

# and append the training set
no_test = np.ones((84,512))
no_test[0:48,:] = labeled_train
no_test[48:84,:] = ambig_feat


# same parameters as the baseline SVM
svm_retrain = SVC(C=1,kernel='linear')
svm_retrain.fit(no_test, label_list_retrain)

# make a new prediction on the initially unlabeled samples
unknown_labels_again = svm_retrain.predict(ambig_feat)

# if there is no change, this value should be 0 and we are done
convergence_check = np.amax(abs(unknown_labels - unknown_labels_again))==0
iterations = 1

# if not done, this loop will run until convergence is reached
while convergence_check == False:
    unknown_labels = unknown_labels_again[:]
    label_list_retrain = label_list_train + list(unknown_labels)
    svm_retrain = SVC(C=1,kernel='linear')
    svm_retrain.fit(no_test, label_list_retrain)
    unknown_labels_again = svm_retrain.predict(ambig_feat)
    convergence_check = np.amax(abs(unknown_labels - unknown_labels_again))==0
    iterations = iterations + 1

self_train_labels = unknown_labels_again[:]

