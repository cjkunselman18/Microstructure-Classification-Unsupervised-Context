library(M3C)
library(ConsensusClusterPlus)

# Read in excel sheet from python and view
library(readxl)
Data_Average_Pooling <- read_excel("Data_Average_Pooling.xlsx", col_names = FALSE)
View(Data_Average_Pooling)

# turns the data frame into a matrix
data <- data.matrix(Data_Average_Pooling)

# run M3C to determine optimal K; 100 iterations means 100 reference distributions,
# and consensus clustering is happening 100 times for each (default values)
# the internal clustering method is K-means with euclidean distance
results <- M3C(data, cores = 2, iters = 100, maxK = 15, seed = 18, clusteralg = 'km')

# the resulting RCSI and p-score plots tell you where to look next; K = 7 has the 
# highest RCSI with a significant p-score

# Now we just run Consensus Clustering with 1000 iterations to get more accurate
# cluster assignments
ccp_res <- ConsensusClusterPlus(d = data, maxK = 15, reps = 1000, pItem = 0.8, clusterAlg = 'km', distance = 'euclidean', seed = 18)

# and we calculate cluster and item consensus values
icl <- calcICL(ccp_res)

# Now we need to do a sanity check to ensure K = 7 makes sense
# Firstly, we check the size of each cluster to make sure that none 
# are too small (say less than 5) to be useful for training a classifier with
sum(ccp_res[[7]]$consensusClass==1) > 5
sum(ccp_res[[7]]$consensusClass==2) > 5
sum(ccp_res[[7]]$consensusClass==3) > 5
sum(ccp_res[[7]]$consensusClass==4) > 5
sum(ccp_res[[7]]$consensusClass==5) > 5
sum(ccp_res[[7]]$consensusClass==6) > 5
sum(ccp_res[[7]]$consensusClass==7) > 5

# All true, so we move on to looking at the images; the code below sorts the images
# based on item consensus for each discovered class - look at the top 3 and make
# sure they all look the same and that no classes are redundant

consensus = icl$itemConsensus$itemConsensus[icl$itemConsensus$k==7]
consensus1 = consensus[1:100]
consensus2 = consensus[(1*100+1):(2*100)]
consensus3 = consensus[(2*100+1):(3*100)]
consensus4 = consensus[(3*100+1):(4*100)]
consensus5 = consensus[(4*100+1):(5*100)]
consensus6 = consensus[(5*100+1):(6*100)]
consensus7 = consensus[(6*100+1):(7*100)]

argsort1 = sort(consensus1,decreasing=TRUE,index.return=TRUE)
argsort2 = sort(consensus2,decreasing=TRUE,index.return=TRUE)
argsort3 = sort(consensus3,decreasing=TRUE,index.return=TRUE)
argsort4 = sort(consensus4,decreasing=TRUE,index.return=TRUE)
argsort5 = sort(consensus5,decreasing=TRUE,index.return=TRUE)
argsort6 = sort(consensus6,decreasing=TRUE,index.return=TRUE)
argsort7 = sort(consensus7,decreasing=TRUE,index.return=TRUE)

items = icl$itemConsensus$item[icl$itemConsensus$k==7]
items1 = items[1:100]
items2 = items[(1*100+1):(2*100)]
items3 = items[(2*100+1):(3*100)]
items4 = items[(3*100+1):(4*100)]
items5 = items[(4*100+1):(5*100)]
items6 = items[(5*100+1):(6*100)]
items7 = items[(6*100+1):(7*100)]

# Now you have everything that you need for the sanity check. For class 1, the 
# first three images are given by
items1[argsort1$ix[1:3]]
# this looks good, and when you view the rest, so do they!
items2[argsort2$ix[1:3]]
items3[argsort3$ix[1:3]]
items4[argsort4$ix[1:3]]
items5[argsort5$ix[1:3]]
items6[argsort6$ix[1:3]]
items7[argsort7$ix[1:3]]

# Now we just need to organize and export this data back to python. We will be
# taking the bottom third of each class to be ambiguous
class1 = which(ccp_res[[7]]$consensusClass==1)
class2 = which(ccp_res[[7]]$consensusClass==2)
class3 = which(ccp_res[[7]]$consensusClass==3)
class4 = which(ccp_res[[7]]$consensusClass==4)
class5 = which(ccp_res[[7]]$consensusClass==5)
class6 = which(ccp_res[[7]]$consensusClass==6)
class7 = which(ccp_res[[7]]$consensusClass==7)

items1_numeric = c()
for (i in 1:100){
  the_str = as.character(items1[i])
  the_str = substr(the_str,4,nchar(the_str))
  items1_numeric[i] = as.numeric(the_str)
}

class1argsort = sort(consensus1[match(class1,items1_numeric)],decreasing = TRUE, index.return = TRUE)
class1_labeled = class1[class1argsort$ix[1:floor(2*length(class1)/3)]]
class1_ambiguous = class1[class1argsort$ix[(floor(2*length(class1)/3)+1):length(class1)]]

items2_numeric = c()
for (i in 1:100){
  the_str = as.character(items2[i])
  the_str = substr(the_str,4,nchar(the_str))
  items2_numeric[i] = as.numeric(the_str)
}

class2argsort = sort(consensus2[match(class2,items2_numeric)],decreasing = TRUE, index.return = TRUE)
class2_labeled = class2[class2argsort$ix[1:floor(2*length(class2)/3)]]
class2_ambiguous = class2[class2argsort$ix[(floor(2*length(class2)/3)+1):length(class2)]]

items3_numeric = c()
for (i in 1:100){
  the_str = as.character(items3[i])
  the_str = substr(the_str,4,nchar(the_str))
  items3_numeric[i] = as.numeric(the_str)
}

class3argsort = sort(consensus3[match(class3,items3_numeric)],decreasing = TRUE, index.return = TRUE)
class3_labeled = class3[class3argsort$ix[1:floor(2*length(class3)/3)]]
class3_ambiguous = class3[class3argsort$ix[(floor(2*length(class3)/3)+1):length(class3)]]

items4_numeric = c()
for (i in 1:100){
  the_str = as.character(items4[i])
  the_str = substr(the_str,4,nchar(the_str))
  items4_numeric[i] = as.numeric(the_str)
}

class4argsort = sort(consensus4[match(class4,items4_numeric)],decreasing = TRUE, index.return = TRUE)
class4_labeled = class4[class4argsort$ix[1:floor(2*length(class4)/3)]]
class4_ambiguous = class4[class4argsort$ix[(floor(2*length(class4)/3)+1):length(class4)]]

items5_numeric = c()
for (i in 1:100){
  the_str = as.character(items5[i])
  the_str = substr(the_str,4,nchar(the_str))
  items5_numeric[i] = as.numeric(the_str)
}

class5argsort = sort(consensus5[match(class5,items5_numeric)],decreasing = TRUE, index.return = TRUE)
class5_labeled = class5[class5argsort$ix[1:floor(2*length(class5)/3)]]
class5_ambiguous = class5[class5argsort$ix[(floor(2*length(class5)/3)+1):length(class5)]]

items6_numeric = c()
for (i in 1:100){
  the_str = as.character(items6[i])
  the_str = substr(the_str,4,nchar(the_str))
  items6_numeric[i] = as.numeric(the_str)
}

class6argsort = sort(consensus6[match(class6,items6_numeric)],decreasing = TRUE, index.return = TRUE)
class6_labeled = class6[class6argsort$ix[1:floor(2*length(class6)/3)]]
class6_ambiguous = class6[class6argsort$ix[(floor(2*length(class6)/3)+1):length(class6)]]

items7_numeric = c()
for (i in 1:100){
  the_str = as.character(items7[i])
  the_str = substr(the_str,4,nchar(the_str))
  items7_numeric[i] = as.numeric(the_str)
}

class7argsort = sort(consensus7[match(class7,items7_numeric)],decreasing = TRUE, index.return = TRUE)
class7_labeled = class7[class7argsort$ix[1:floor(2*length(class7)/3)]]
class7_ambiguous = class7[class7argsort$ix[(floor(2*length(class7)/3)+1):length(class7)]]

ambiguous = c(class1_ambiguous,class2_ambiguous,class3_ambiguous,class4_ambiguous,class5_ambiguous,class6_ambiguous,class7_ambiguous)

library(openxlsx)

wb = createWorkbook()
addWorksheet(wb,'Sheet1')
writeData(wb,1,class1_labeled,startCol = 1)
writeData(wb,1,class2_labeled,startCol = 2)
writeData(wb,1,class3_labeled,startCol = 3)
writeData(wb,1,class4_labeled,startCol = 4)
writeData(wb,1,class5_labeled,startCol = 5)
writeData(wb,1,class6_labeled,startCol = 6)
writeData(wb,1,class7_labeled,startCol = 7)
writeData(wb,1,ambiguous,startCol = 8)
saveWorkbook(wb,'discovered_classes.xlsx',overwrite=TRUE)
