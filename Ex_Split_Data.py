# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:48:18 2020

@author: Courtney
"""

import numpy as np
import xlrd

# Extract label information from excel spreadsheet
book = xlrd.open_workbook("discovered_classes.xlsx")

sheet = book.sheet_by_index(0)

class_1_ims = []

for i in range(0,10):
    class_1_ims.append(sheet.cell_value(i,0))
    
class_1_ims.sort()

class_2_ims = []

for i in range(0,10):
    class_2_ims.append(sheet.cell_value(i,1))
    
class_2_ims.sort()
    
class_3_ims = []

for i in range(0,9):
    class_3_ims.append(sheet.cell_value(i,2))
    
class_3_ims.sort()
    
class_4_ims = []

for i in range(0,8):
    class_4_ims.append(sheet.cell_value(i,3))
    
class_4_ims.sort()
    
class_5_ims = []

for i in range(0,12):
    class_5_ims.append(sheet.cell_value(i,4))
    
class_5_ims.sort()

class_6_ims = []

for i in range(0,7):
    class_6_ims.append(sheet.cell_value(i,5))
    
class_6_ims.sort()

class_7_ims = []

for i in range(0,8):
    class_7_ims.append(sheet.cell_value(i,6))
    
class_7_ims.sort()

ambig_ims = []

for i in range(0,36):
    ambig_ims.append(sheet.cell_value(i,7))
    
ambig_ims.sort()


# Split feature vectors into classes        
class_1_feat = layer_5_matrix_avg[np.isin(range(0,100),(np.array(class_1_ims)-1)),:]
class_2_feat = layer_5_matrix_avg[np.isin(range(0,100),(np.array(class_2_ims)-1)),:]
class_3_feat = layer_5_matrix_avg[np.isin(range(0,100),(np.array(class_3_ims)-1)),:]
class_4_feat = layer_5_matrix_avg[np.isin(range(0,100),(np.array(class_4_ims)-1)),:]
class_5_feat = layer_5_matrix_avg[np.isin(range(0,100),(np.array(class_5_ims)-1)),:]
class_6_feat = layer_5_matrix_avg[np.isin(range(0,100),(np.array(class_6_ims)-1)),:]
class_7_feat = layer_5_matrix_avg[np.isin(range(0,100),(np.array(class_7_ims)-1)),:]
ambig_feat = layer_5_matrix_avg[np.isin(range(0,100),(np.array(ambig_ims)-1)),:]


# Split labeled data into training and testing sets
from sklearn.model_selection import train_test_split


label_list = []

for i in range(0,len(class_1_ims)):
    label_list.append(1)
for i in range(0,len(class_2_ims)):
    label_list.append(2)
for i in range(0,len(class_3_ims)):
    label_list.append(3)
for i in range(0,len(class_4_ims)):
    label_list.append(4)
for i in range(0,len(class_5_ims)):
    label_list.append(5)
for i in range(0,len(class_6_ims)):
    label_list.append(6)
for i in range(0,len(class_7_ims)):
    label_list.append(7)

image_number_list = class_1_ims + class_2_ims + class_3_ims + class_4_ims + class_5_ims + class_6_ims + class_7_ims

labeled_data = np.ones((64,512))
labeled_data[0:len(class_1_ims),:] = class_1_feat
labeled_data[len(class_1_ims):(len(class_1_ims) + len(class_2_ims)),:] = class_2_feat
labeled_data[(len(class_1_ims)+len(class_2_ims)):(len(class_1_ims) + len(class_2_ims) + len(class_3_ims)),:] = class_3_feat
labeled_data[(len(class_1_ims)+len(class_2_ims) + len(class_3_ims)):(len(class_1_ims) + len(class_2_ims) + len(class_3_ims) + len(class_4_ims)),:] = class_4_feat
labeled_data[(len(class_1_ims)+len(class_2_ims) + len(class_3_ims) + len(class_4_ims)):(len(class_1_ims) + len(class_2_ims) + len(class_3_ims) + len(class_4_ims) + len(class_5_ims)),:] = class_5_feat
labeled_data[(len(class_1_ims)+len(class_2_ims) + len(class_3_ims) + len(class_4_ims) + len(class_5_ims)):(len(class_1_ims) + len(class_2_ims) + len(class_3_ims) + len(class_4_ims) + len(class_5_ims) + len(class_6_ims)),:] = class_6_feat
labeled_data[(len(class_1_ims)+len(class_2_ims) + len(class_3_ims) + len(class_4_ims) + len(class_5_ims) + len(class_6_ims)):(len(class_1_ims) + len(class_2_ims) + len(class_3_ims) + len(class_4_ims) + len(class_5_ims) + len(class_6_ims) + len(class_7_ims)),:] = class_7_feat

labeled_train, labeled_test, label_list_train, label_list_test, image_number_train, image_number_test = train_test_split(labeled_data,label_list,image_number_list,test_size=.25,random_state=2018)
