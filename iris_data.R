install.packages("xgboost")
install.packages("readr")
library(xgboost)
library(readr)
library(stringr)
library(caret)

setwd("E:/Analytics/Analytics Vidhya Datahack/24 Case Study Challenge")
iris.data<-read.table("iris.data.txt",sep = "," )
colnames(iris.data)<-c("sepal.length","sepal.width","petal.length","petal.width","class")

#randomize and reindex data
iris.data<-iris.data[sample(nrow(iris.data)),]
row.names(iris.data) <- 1:nrow(iris.data)

#creating data sets
partition_1 = 0.6*nrow(iris.data)
iris.data.train<-iris.data[1:partition_1,]
iris.data.test<-iris.data[91:150,]

#run multiclass classification
#convert labels using min-max scaling
labels<-as.numeric(iris.data.train$class)-1
ts_labels<-as.numeric(iris.data.test$class)-1
#convert datasets
train<-model.matrix(~.+0,data=iris.data.train[,-5])
test<-model.matrix(~.+0,data=iris.data.test[,-5])
#xgboost

#preparing matrix
dtrain<-xgb.DMatrix(data = train,label=labels)
dtest<-xgb.DMatrix(data = test,label=ts_labels)

#default parameters
params <- list(booster = "gbtree", objective = "multi:softprob", num_class  = 3, eval_metric = "mlogloss")
xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, nfold = 5, showsd = T, 
                 stratified = T, print_every_n = 10, early_stop_round = 20, maximize = F,prediction = T)

# Mutate xgb output to deliver hard predictions  
OOF_prediction <- data.frame(xgbcv$pred) %>%
  dplyr::mutate(max_prob = max.col(., ties.method = "last"),label = labels + 1)
head(OOF_prediction)

#confusion matrix
confusionMatrix(factor(OOF_prediction$max_prob),
                factor(OOF_prediction$label),
                mode = "everything")

#fit the model on train data    
xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 100)

#evaluate
xgbpred <- predict (xgb1,dtest)
test_prediction <- matrix(xgbpred, nrow = 3,
                          ncol=length(xgbpred)/3) %>%
  t() %>%
  data.frame() %>%
  dplyr::mutate(label = ts_labels + 1,
         max_prob = max.col(., "last"))

# confusion matrix of test set
confusionMatrix(factor(test_prediction$max_prob),
                factor(test_prediction$label),
                mode = "everything")

