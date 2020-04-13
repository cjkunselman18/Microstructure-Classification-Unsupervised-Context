# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:54:04 2020

@author: Courtney
"""

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import xlsxwriter

# Import trained VGG16 architecture
base_model = VGG16(weights='imagenet',include_top=False)

# We want the 3rd layer of the last convolutional block, C5,3
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv3').output)

# now we take the average of each filter slice
layer_5_list_avg = []

for i in range(1,101):
    path = 'Example Images\\Ex_%s.jpg' %(int(i))
    img = image.load_img(path)  # size of image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    layer = model.predict(x) # the full layer
    feature_vector = np.mean(layer, axis = tuple([0,1,2]))  # the average
    layer_5_list_avg.append(feature_vector)
    
layer_5_matrix_avg = np.ones((100,512))
for i in range(0,100):
    layer_5_matrix_avg[i,:] = layer_5_list_avg[i]


# Now we need to export this data for Consensus Clustering in R
workbook = xlsxwriter.Workbook('Data_Average_Pooling.xlsx')
worksheet = workbook.add_worksheet()


row = 1
for col, data in enumerate(layer_5_matrix_avg):
    worksheet.write_column(row, col, data)


workbook.close()




    


