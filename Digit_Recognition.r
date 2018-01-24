start<- Sys.time()
############################ SVM Digit Recogniser #################################
# 1. Business Understanding
# 2. Data Understanding
# 3. Data Preparation
# 4. Model Building 
#  4.1 Linear kernel
#  4.2 RBF Kernel
# 5 Hyperparameter tuning and cross validation

#####################################################################################

# 1. Business Understanding: 

#The objective is to identify each of a large number of black-and-white
#rectangular pixel displays as one of the digits from 0 to 9

#####################################################################################

# 2. Data Understanding: 
#Have been given 2 set of data- train and test
#train-data 
# Number of Instances: 60,000
# Number of Attributes: 785
#test-data 
# Number of Instances: 10,000
# Number of Attributes: 785

#######################################################################################
#3. Data Preparation: 


#Loading Neccessary libraries

library(kernlab)
library(readr)
library(caret)
library(dplyr)
library(h2o)

#Loading train and test data

train <- read.csv("mnist_train.csv",stringsAsFactors = F,header = F)
test <- read.csv("mnist_test.csv",stringsAsFactors = F,header = F)

#Understanding Dimensions

dim(train) ##60000   785

dim(test) ##10000   785

#Structure of train and test dataset

str(train)
str(test)

#printing first few rows of train and test dataset

head(train)
head(test)

#Exploring train and test dataset

summary(train)
summary(test)

#checking missing value in train and test dataset

sum(is.na(train))  ## No NA present
sum(is.na(test))   ##No NA present

##As we have a lot of columns(785). we will use PCA to reduce the columns
## The way to do it like, merge train and test dataset, apply PCA , post that again split the merged sheet into train and test data set 
merged_dataset <- rbind(train,test) ##merging train and test data 
dim(merged_dataset)  ##70000   785
##Removing target variable(digit)
digit_features <- merged_dataset[,-1]
##Checking for variance
nzv_obj <- nearZeroVar(digit_features, saveMetrics = T)
##Removing those columns having zeroVar value as False and creating a new dataframe
digit_features2 <- digit_features[,nzv_obj$zeroVar==F]
dim(digit_features2) ##70000   719

##Scaling the features
digit_features2 <- scale(digit_features2)
##Taking a threshold of 70%, applying PCA to get relevant columns 
preobj <- preProcess(digit_features2, method=c("pca"), thresh = 0.70)
digit_features3<- predict(preobj, digit_features2)
dim(digit_features3)   ##70000    99
digit_features3 <- cbind(merged_dataset$V1,digit_features3)
digit_features3 <- data.frame(digit_features3)
colnames(digit_features3)[1] <- "digit"
digit_features3$digit <- factor(digit_features3$digit)

 ##Separating train and test dataset
 processed_train <- head(digit_features3,60000)
 dim(processed_train) ##60000   100
 
 processed_test <- tail(digit_features3,10000)
 dim(processed_test) ##10000   100
 
 
 ##Taking a sample of 15000 records(25%) data from train dataset
sample_train <- processed_train%>%
group_by(digit)%>%
sample_n(1500,replace = T)
summary(sample_train$digit)
#Creating a validation dataset to be used by neural network for tuning
subsamplesize <- 15000
subsample <- sample(nrow(processed_train), subsamplesize, replace=FALSE)
validationsize <- subsamplesize*0.3
validationsample <- subsample[1:validationsize]
mnistValidation <- processed_train[validationsample,]

##Taking a sample of 2500 records(25%) data from test dataset
sample_test <- processed_test%>%
group_by(digit)%>%
sample_n(250,replace = T)
summary(sample_test$digit)
##Writing train,test,validation data set to working directory
write.csv(sample_train, file='sample_train.csv', row.names=FALSE)
write.csv(mnistValidation, file='mnist-validation.csv', row.names=FALSE)
write.csv(sample_test, file='sample_test.csv', row.names=FALSE)

#######################################################################################

##4.Constructing Model using SVM 

#4.1Using Linear Kernel
Model_linear <- ksvm(digit~ ., data = sample_train, scale = FALSE, kernel = "vanilladot")
Eval_linear<- predict(Model_linear, sample_test)
#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear,sample_test$digit)
#Accuracy : 0.9248

#4.2Using RBF Kernel
Model_RBF <- ksvm(digit~ ., data = sample_train, scale = FALSE, kernel = "rbfdot")
Eval_RBF<- predict(Model_RBF, sample_test)

#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF,sample_test$digit)

#Accuracy : 0.9464    RBF model is better than linear model 

############   Hyperparameter tuning and Cross Validation #####################

# We will use the train function from caret package to perform Cross Validation. 

#traincontrol function Controls the computational nuances of the train function.
# i.e. method =  CV means  Cross Validation.
#      Number = 5 implies Number of folds in CV.
trainControl <- trainControl(method="cv", number=5)


# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.

metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.

set.seed(7)
grid <- expand.grid(.sigma=seq(0.01,0.05,by=0.02), .C=seq(0.2,1,by=0.2) )


#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

fit.svm <- train(digit~., data=sample_train, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)
				

print(fit.svm)

##The final values used for the model were sigma = 0.01 and C = 1.  Accuracy = 0.9464000 
##Evaluating the sample test dadaset

plot(fit.svm)
Eval_svm<- predict(fit.svm, sample_test)
confusionMatrix(Eval_svm,sample_test$digit)

end <- Sys.time()
end-start

#Accuracy : 0.9456
#Sensitivity and specificity for each digit prediction is as given below
 #                     Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6  Class: 7 Class: 8 Class: 9
#Sensitivity            0.9680   0.9920   0.9480   0.9600   0.9560   0.9280   0.9480     0.9240   0.9000   0.9320
#Specificity            0.9982   0.9969   0.9880   0.9938   0.9960   0.9898   0.9964     0.9938   0.9929   0.9938

##This is a good model, as accuracy,sensitivity and specificity is around 95 % on test dadaset.
##----------------------------------------------------------------

##Applying neural network on same dataset with specs as 
#Activation function: RectifierWithDropout
#Neurons in the mid layers: 200
#dropout_ratio :0.1
#epochs: 20
# Initialize the h2o environment
library(h2o)
h2o.init()

mnistTrain <- h2o.importFile("sample_train.csv")
mnistValidation <- h2o.importFile("mnist-validation.csv")
mnistTest <- h2o.importFile("sample_test.csv")
# Convert the first column to a factor
digitlabel <- "digit"
pixels <- setdiff(names(mnistTrain), digitlabel)
mnistTrain[,digitlabel] <- as.factor(mnistTrain[,digitlabel])
mnistValidation[,digitlabel] <- as.factor(mnistValidation[,digitlabel])

set.seed(1105)
# Perform 5-fold cross-validation on the training_frame
mnistNet <- h2o.deeplearning(x = pixels,
                             y = digitlabel,
                             training_frame = mnistTrain,
                             validation_frame = mnistValidation,
                             distribution = "multinomial",
                             activation = "RectifierWithDropout",
                             hidden = c(200,200,200),
                             hidden_dropout_ratio = c(0.1, 0.1, 0.1),
                             l1 = 1e-5,
                             epochs = 20)

# Test the model on the test data
mnistTest[,digitlabel] <- as.factor(mnistTest[,digitlabel])
digitPrediction <- h2o.predict(mnistNet, mnistTest)
ImageId <- as.numeric(seq(1,nrow(mnistTest)))
names(ImageId)[1] <- "ImageId"
predictions <- cbind(as.data.frame(ImageId),as.data.frame(digitPrediction[,1]))
names(predictions)[2] <- "Label"
write.table(as.matrix(predictions), file="DNN_pred.csv", row.names=FALSE, sep=",")
predict_val <- read.csv("DNN_pred.csv",stringsAsFactors =F)
test1 <- read.csv("sample_test.csv",stringsAsFactors = F)
predict_val$actuallabel <- test1$digit
confusionMatrix(predict_val$actuallabel,predict_val$Label)
##Accuracy : 0.8944 
#Statistics by Class:

#                     Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
#Sensitivity            0.8799   0.9690   0.9545   0.8897   0.9902   0.9830   0.9955   0.7585   0.7335   0.9541
#Specificity            0.9995   1.0000   0.9916   0.9928   0.9795   0.9669   0.9864   0.9977   0.9977   0.9727




                      