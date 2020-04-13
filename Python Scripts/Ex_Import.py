# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:01:37 2020

@author: Courtney
"""

import xlrd


# read in new labels
book = xlrd.open_workbook("S4VM_Label_Prediction_Example.xlsx")
sheet = book.sheet_by_index(0)
s4vm_labels = []
for i in range(0,36):
    s4vm_labels.append(sheet.cell_value(i+1,0))
    
    
