# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:32:04 2020

@author: Courtney
"""

import xlsxwriter

workbook = xlsxwriter.Workbook('Data_Export_S4VM.xlsx')
worksheet = workbook.add_worksheet()

for i in range(1,512):
    worksheet.write(0,i-1, 'Slice %s' % (i))

row = 1
for col, data in enumerate(no_test.T):
    worksheet.write_column(row, col, data)

workbook.close()

workbook = xlsxwriter.Workbook('Labels_Export_S4VM.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write(0,0, 'labels')
worksheet.write_column(1,0, np.array(label_list_train).T)


workbook.close()
