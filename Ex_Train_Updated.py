# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:59:44 2020

@author: Courtney
"""
    
# find where they all agree
unknowns_add_to_training = []
unknown_labels_for_training = []
for i in range(0,36):
    for j in range(1,5):
        if label_propagation_unknown_labels[i] == j and self_train_labels[i] == j and clusters[i+48] == j and s4vm_labels[i] == j:
            unknowns_add_to_training.append(i)
            unknown_labels_for_training.append(j)
    

    

# then we add this subset with correpsonding labels to the training set        
training_with_unknowns = np.ones((48+len(unknowns_add_to_training),512))
training_with_unknowns[0:48,:] = labeled_train
training_with_unknowns[48:(48+len(unknowns_add_to_training)),:] = ambig_feat[unknowns_add_to_training,:]

# Now we optimize hyperparameters and train the updated SVM as we did the baseline SVM
training_with_unknowns_labels = label_list_train + unknown_labels_for_training


print("# Tuning hyper-parameters for accuracy")
print()

clf = GridSearchCV(SVC(), tuned_parameters, cv=cv, scoring='accuracy' )
clf.fit(training_with_unknowns, training_with_unknowns_labels)
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

    

# Test accuracy
updated_svm = clf.best_estimator_
predicted_test_labels_updated = updated_svm.predict(labeled_test)
mat = confusion_matrix(label_list_test, predicted_test_labels_updated)
plt.figure(0)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('True Label')
plt.ylabel('Predicted Label')

