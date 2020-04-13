# Microstructure Classification in the Unsupervised Context

This example uses 100 randomly-selected images from our dataset of 1925. The data is featurized using the last layer of the fifth convolutional layer of the VGG16 architecture, and the dimensionality of the space is reduced using average pooling. M3C using K-means clustering is then run on this data to determine the optimal cluster number. Item consensus values are used to determine high-confidence and ambiguous data, and this data is used to train the classifiers in the semi-supervised framework. All data shared between python/R/Matlab is in the form of Excel documents.

Necessary Packages (beyond base):

Python:

•	Keras

•	Numpy

•	Xlsxwriter

•	Xlrd

•	Sklearn

•	Seaborn

•	Matplotlib

•	Copkmeans (https://github.com/Behrouz-Babaki/COP-Kmeans)

•	Random

R:

•	M3C (you will need BiocManager to install)

•	ConsensusClusterPlus (need BiocManager)

•	Openxlsx

Matlab:

•	S4VM (http://lamda.nju.edu.cn/code_S4VM.ashx?AspxAutoDetectCookieSupport=1)

INSTRUCTIONS:

1.	Download all of the files into the same folder and set this as your working directory for python, R, and Matlab. An extra folder of Excel files is provided so you can check that you are getting the correct output (or so you can skip steps if you do not have certain software/packages available). The Excel files that you produce will not overwrite those in the Excel Files folder.

2.	Open Ex_Featurization.py and run. This script featurizes all of the images and performs average pooling for dimensionality reduction. It also outputs the Excel file Data_Average_Pooling.xlsx into the working directory.

3.	Open Class_Discovery.R and run. This script reads in Data_Average_Pooling.xlsx and performs consensus clustering and M3C using K-means as the internal clustering method. The M3C implementation provides helpful RCSI and p-score plots (see comments in code for interpretation). Before it is finished, this script also calculates the cluster consensus values (see the variable “icl”) and identifies the high-confidence and ambiguous data using item consensus. The output is an Excel file called discovered_classes.xlsx.

4.	Open Ex_Split_Data.py and run. This script reads in the high-confidence and ambiguous data assignments from discovered_classes.xlsx and splits the high-confidence data into training and validation sets.

5.	Open Ex_Train_Baseline.py and run. This script trains an SVM on only the labeled training data, which will be needed for the Modified Yarowsky and S4VM implementations.

6.	Open Ex_My.py, Ex_LP.py, and Ex_CKM.py and run. These scripts are the implementations of the Modified Yarowsky, Label Propagation, and Cop-KMEANS, and they can be executed in any order. 

7.	Open Ex_Export.py and run. This script outputs Excel files called Data_Export_S4VM.xlsx and Labels_Export_S4VM.xlsx for preparation for implementing S4VM in Matlab.

8.	Open Matlab_S4VM_Example.m and run. This script reads in the training and ambiguous sets, runs S4VM (with beta = 0.9 and mentioned in the report), and outputs an Excel file called S4VM_Label_Prediction_Example.xlsx. Note that the libsvm folder and the S4VM_constant_beta.m functions must be in the working directory for this script to run.

9.	Open Ex_Import.py and run. This script simply reads in the results from the previous step.

10.	Open Ex_Train_Updated.py and run. This script finds the subset of the ambiguous set which received a labeling consensus from the four semi-supervised methods and trains an SVM over the labeled training set and this subset. The validation error results are contained in the confusion matrix.

11.	Open Ex_Train_SS.py and run. This script trains SVMs over the results of each of the semi-supervised methods. Once again, the validation error results are contained in the confusion matrices.

12.	Open Ex_Agreement_Counts.py and run. This script calculates agreement rates among classifiers in preparation for unsupervised error estimation. The output is an Excel file called Agreement Counts Example.xlsx

13.	Open Unsupervised_Error_Example.m and run. This script implements the constrained optimization for unsupervised error approximation. Note that c1.m and Unsupervised_Error_Constraints_Multi_Class.m must be in the working directory for this script to run. The final output is the Excel file Unsupervised_Error_Example.xlsx, which contains the error estimates for the ambiguous sub-population.
