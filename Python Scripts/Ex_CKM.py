# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:08:26 2020

@author: Courtney
"""

from copkmeans.cop_kmeans import cop_kmeans
import random

# We choose one point in each class and create must-link conditions between that
# one point and all others in that class

must_link = []

class_1_sample = np.where(np.array(label_list_train)== 1)[0][0]
class_2_sample = np.where(np.array(label_list_train)== 2)[0][0]
class_3_sample = np.where(np.array(label_list_train)== 3)[0][0]
class_4_sample = np.where(np.array(label_list_train)== 4)[0][0]
class_5_sample = np.where(np.array(label_list_train)== 5)[0][0]
class_6_sample = np.where(np.array(label_list_train)== 6)[0][0]
class_7_sample = np.where(np.array(label_list_train)== 7)[0][0]



for i in range(class_1_sample + 1,48):
    if label_list_train[i] == 1:
        must_link.append((class_1_sample,i))

# This is the same idea       
for i in range(class_2_sample + 1,48):
    if label_list_train[i] == 2:
        must_link.append((class_2_sample,i))

for i in range(class_3_sample + 1,48):
    if label_list_train[i] == 3:
        must_link.append((class_3_sample,i))
        
for i in range(class_4_sample + 1,48):
    if label_list_train[i] == 4:
        must_link.append((class_4_sample,i))
        
for i in range(class_5_sample + 1,48):
    if label_list_train[i] == 5:
        must_link.append((class_5_sample,i))
        
for i in range(class_6_sample + 1,48):
    if label_list_train[i] == 6:
        must_link.append((class_6_sample,i))
        
for i in range(class_7_sample + 1,48):
    if label_list_train[i] == 7:
        must_link.append((class_7_sample,i))
        
# We do not want anything with different labels to be linked
cannot_link = [(class_1_sample,class_2_sample),(class_1_sample,class_3_sample),(class_1_sample,class_4_sample),(class_1_sample,class_5_sample),(class_1_sample,class_6_sample),(class_1_sample,class_7_sample),(class_2_sample,class_3_sample),(class_2_sample,class_4_sample),(class_2_sample,class_5_sample),(class_2_sample,class_6_sample),(class_2_sample,class_7_sample),(class_3_sample,class_4_sample),(class_3_sample,class_5_sample),(class_3_sample,class_6_sample),(class_3_sample,class_7_sample),(class_4_sample,class_5_sample),(class_4_sample,class_6_sample),(class_4_sample,class_7_sample),(class_5_sample,class_6_sample),(class_5_sample,class_7_sample),(class_6_sample,class_7_sample)]

random.seed(18)

# This command runs the method; clusters are the label assignments
clusters, centers = cop_kmeans(dataset=no_test, k=7, ml=must_link,cl=cannot_link)

# remember this is an unsupervised process with no real label information going into it;
# the algorithm will give two clusters, but we have to look into it to see which cluster
# labels correspond to our initial labels

class_1_label = clusters[class_1_sample]
class_2_label = clusters[class_2_sample]
class_3_label = clusters[class_3_sample]
class_4_label = clusters[class_4_sample]
class_5_label = clusters[class_5_sample]
class_6_label = clusters[class_6_sample]
class_7_label = clusters[class_7_sample]


for i in range(0,84):
    if clusters[i] == class_1_label:
        clusters[i] = 1
    elif clusters[i] == class_2_label:
        clusters[i] = 2
    elif clusters[i] == class_3_label:
        clusters[i] = 3
    elif clusters[i] == class_4_label:
        clusters[i] = 4
    elif clusters[i] == class_5_label:
        clusters[i] = 5
    elif clusters[i] == class_6_label:
        clusters[i] = 6
    else:
        clusters[i] = 7
