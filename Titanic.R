test = read.csv("test.csv")
oritest = read.csv("test.csv")
oritest$Survived = NA
train = read.csv("train.csv")
test$Survived = NA
combine = rbind(train, test)
combine$Ticket = NULL
combine$PassengerId = NULL
combine$Cabin = NULL
Sys.setlocale("LC_ALL", "C")
library(mice)
vars = setdiff(names(combine), c("Survived","Name"))
imputed = complete(mice(combine[vars]))
combine[vars] = imputed
combine$Name = as.character(combine$Name)

library(stringr)
combine$title = str_sub(combine$Name, str_locate(combine$Name, ",")[ , 1] + 2, str_locate(combine$Name, "\\.")[ , 1] - 1)
male_noble_names <- c("Capt", "Col", "Don", "Dr", "Jonkheer", "Major", "Rev", "Sir")
combine$title[combine$title %in% male_noble_names] <- "male_noble"
female_noble_names <- c("Lady", "Mlle", "Mme", "Ms", "the Countess")
combine$title[combine$title %in% female_noble_names] <- "female_noble"
combine$title = as.factor(combine$title)
combine$Name=NULL
combine$Survived = as.factor(combine$Survived)

#removing NAs
#tapply(combine$Age, combine$Pclass, mean, na.rm =TRUE)
#combine$Age[is.na(combine$Age) & combine$Pclass==1] = mean(combine$Age[combine$Pclass==1], na.rm = TRUE)
#combine$Age[is.na(combine$Age) & combine$Pclass==2] = mean(combine$Age[combine$Pclass==2], na.rm = TRUE)
#combine$Age[is.na(combine$Age) & combine$Pclass==3] = mean(combine$Age[combine$Pclass==3], na.rm = TRUE)
#combine$Fare[is.na(combine$Fare)] = median(combine$Fare, na.rm = TRUE)
#summary(combine$Age)

#----------randomforest--------
Sys.setlocale("LC_ALL", "C")
library(caret)
library(randomForest)
train = combine[1:891,]
test = combine[892:1309,]
str(train)
RAND = randomForest(Survived~., data=train)
varImpPlot(RAND)
table(train$Survived, RAND$predicted)/nrow(train) #0.58
pred = predict(RAND, newdata = test)
test$Survived = pred
#test$Survived = ifelse(pred>0.5,1,0)
result = oritest[,c("PassengerId", "Survived")]
result$RAND = test$Survived
result$Survived = result$RAND
result[,1:2]
write.csv(result[,1:2], "result.csv", row.names = F)


#-------------SVM-----------------
combine$Survived = as.factor(combine$Survived)

library(caret)
library(e1071)
train = combine[1:891,]
test = combine[892:1309,]
preproc = preProcess(train[,-1])
trainNorm = predict(preproc, train)
testNorm = predict(preproc, test)

tmodel = tune(svm, Survived~., data = trainNorm, ranges = list(epsilon = seq(0,1,0.1), cost = 2^(2:5)))
model = tmodel$best.model
confusionMatrix(table(train$Survived , model$fitted))

summary(predTest)
testNorm$Survived=NULL
predTest = predict(model, newdata = testNorm)
test$Survived = predTest
result = oritest[,c("PassengerId", "Survived")]
result$SVM = test$Survived
table(result$RAND, result$SVM)
write.csv(result[,1:2], "result.csv", row.names = F)

#----------------Logistic-------------------
combine$Survived = as.factor(combine$Survived)
combine$Embarked = NULL
library(caret)
train = combine[1:891,]
test = combine[892:1309,]
log = glm(Survived~., data=train, family = "binomial")
summary(test)
predLog = predict(log, newdata = test, type="response")
test$Survived = ifelse(predLog>0.5,1,0)
table(test$Survived)
result = oritest[,c("PassengerId", "Survived")]
result$Log = test$Survived
result$Survived = result$Log
write.csv(result[,1:2], "result.csv", row.names = F)

#------------------XGboost------------------

Sys.setlocale("LC_ALL", "C")
library(data.table)
library(mlr)
library(xgboost)

#convert factors to numeric from 0
str(combine)
summary(combine)
combine$Survived = as.numeric(combine$Survived) - 1
combine$Sex = as.numeric(combine$Sex) - 1
combine$Embarked = as.numeric(combine$Embarked) - 1
combine$title = as.numeric(combine$title) - 1
combine$Pclass = as.numeric(combine$Pclass) - 1

train = combine[1:891,]
test = combine[892:1309,]
str(train)


#convert data frame to data table
setDT(train) 
setDT(test)

table(is.na(train))
sapply(train, function(x) sum(is.na(x))/length(x))*100
table(is.na(test))
sapply(test, function(x) sum(is.na(x))/length(x))*100

#set all missing value as "Missing" 
train[is.na(train)] <- "Missing" 
test[is.na(test)] <- "Missing"

#using one hot encoding 
labels <- train$Survived 
ts_label <- test$Survived
new_tr <- model.matrix(~.+0,data = train[,-c("Survived"),with=F]) 
new_ts <- model.matrix(~.+0,data = test[,-c("Survived"),with=F])

#convert factor to numeric 
labels = as.numeric(labels)
ts_label = as.numeric(ts_label)

#preparing matrix 
dtrain <- xgb.DMatrix(data = new_tr, label = labels) 
dtest <- xgb.DMatrix(data = new_ts, label = ts_label)

#default parameters
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.1, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)

xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, nfold = 3, showsd = T, stratified = T, print.every.n = 10, early.stop.round = 20, maximize = F)
##best iteration = 31

min(xgbcv[["evaluation_log"]][["test_error_mean"]])

#first default - model training
xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 31, watchlist = list(val=dtest,train=dtrain), print.every.n = 10, early.stop.round = 10, maximize = F , eval_metric = "error")
#model prediction
xgbpred <- predict (xgb1,dtest)
xgbpred <- ifelse (xgbpred > 0.5,1,0)

#view variable importance plot
mat <- xgb.importance (feature_names = colnames(new_tr),model = xgb1)
xgb.plot.importance (importance_matrix = mat[1:20]) 



#convert characters to factors
fact_col <- colnames(train)[sapply(train,is.character)]

for(i in fact_col) set(train,j=i,value = factor(train[[i]]))
for(i in fact_col) set(test,j=i,value = factor(test[[i]]))

train$Survived = as.factor(train$Survived)
test$Survived = as.factor(test$Survived)

#create tasks
traintask <- makeClassifTask (data = train, target = "Survived")
testtask <- makeClassifTask (data = test, target = "Survived")

#do one hot encoding`<br/> 
traintask <- createDummyFeatures(obj = traintask) 
testtask <- createDummyFeatures(obj = testtask)

#create learner
lrn <- makeLearner("classif.xgboost",predict.type = "response")
lrn$par.vals <- list( objective="binary:logistic", eval_metric="error", nrounds=1000L, eta=0.01)

#set parameter space
params <- makeParamSet( makeDiscreteParam("booster",values = c("gbtree","gblinear")), makeIntegerParam("max_depth",lower = 3L,upper = 10L), makeNumericParam("min_child_weight",lower = 1L,upper = 10L), makeNumericParam("subsample",lower = 0.5,upper = 1), makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))

#set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)

#search strategy
ctrl <- makeTuneControlRandom(maxit = 10L)

#set parallel backend
library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())

#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params, control = ctrl, show.info = T)
mytune$y 
#0.8712878

#set hyperparameters
lrn_tune = setHyperPars(lrn, par.vals = mytune$x)

#train model
xgmodel = train(learner = lrn_tune, task = traintask)

#predict model
xgpred <- predict(xgmodel,testtask)

result = oritest[,c("PassengerId", "Survived")]
result$Survived = xgpred$data$response
summary(result)

write.csv(result,"result.csv", row.names = F)
