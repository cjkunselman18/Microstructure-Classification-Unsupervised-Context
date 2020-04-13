% read in data from excel document
data = readtable('Data_Export_S4VM.xlsx');
labels = readtable('Labels_Export_S4VM.xlsx');

% the above command turns the data into tables -- use this to get arrays
data = table2array(data);
labels = table2array(labels);

% get inputs from read-in data
X_train = data(1:48,:);
X_unknown = data(49:84,:);

% parameters - C1 is from the baseline SVM, and the rest are defaults
% (using linear kernel since that was what was chosen for baseline)
C1 = 1;
C2 = 0.1;
gamma = 0;
sampleTime = 100;

% run S4VM; "prediction_unlabeled" is the predicted labels for the unlabeled set
addpath('libsvm-mat-2.89-3-box constraint');
prediction_unlabeled = S4VM_constant_beta(X_train,labels,X_unknown,'Linear',C1,C2,sampleTime,gamma);

% write to new excel document so we can load these results back into python
prediction_table = array2table(prediction_unlabeled)
filename = 'S4VM_Label_Prediction_Example.xlsx'
writetable(prediction_table,filename)