### The first technique 
getwd()
setwd("/Users/siddharthsaxena")

install.packages("magrittr") 
install.packages("dplyr")    
library(magrittr) 
library(dplyr)  
#Import data
mydata<-read.csv('../diabetic_data.csv')

#Check NULL value
is.null(mydata)
#Delete missing value
mydata<-na.omit(mydata)

#Drop rows with value '?'
i<-1
for(i in 1:50){
  mydata<-mydata[!(mydata[,i]=='?'),]
}

# Turn columns diag_1, diag_2, diag_3 into binary variable
# Becasue code 250 represents diabetic, when value is 250 fill 1 otherwise fill 0
mydata$diag_1<-round(as.numeric(mydata$diag_1))
mydata$diag_2<-round(as.numeric(mydata$diag_2))
mydata$diag_3<-round(as.numeric(mydata$diag_3))

for(i in 1:1043){
  if(mydata[i,19]==250){
    mydata[i,19]<-1
  }else{
    mydata[i,19]<-0
  }
}

for(i in 1:1043){
  if(mydata[i,20]==250){
    mydata[i,20]<-1
  }else{
    mydata[i,20]<-0
  }
}

for(i in 1:1043){
  if(mydata[i,21]==250){
    mydata[i,21]<-1
  }else{
    mydata[i,21]<-0
  }
}

#Drop columns that will not be used
mydata<-subset(mydata, select = c(time_in_hospital, num_lab_procedures, num_procedures, number_outpatient,
                                  number_emergency, number_inpatient, diag_1, diag_2, diag_3,
                                  number_diagnoses, readmitted))


#Divide data set into train data and test data
train_data<-mydata%>%sample_frac(.7)
test_data<-mydata%>%sample_frac(.3)

#Export train data and test data
write.csv(train_data, '/Users/siddharthsaxena/Desktop/knn_model_train_data.csv', row.names = FALSE)
write.csv(test_data, '/Users/siddharthsaxena/Desktop/knn_model_test_data.csv', row.names = FALSE)


#Transform data into suitable form for modeling
data_norm<-function(x) {
  ((x - min(x))/ (max(x)- min(x)))
}

j<-1
for(j in 1:10){
  train_data[,j]<-as.numeric(as.character(train_data[,j]))
  test_data[,j]<-as.numeric(as.character(test_data[,j]))
}

#The column 7 to 9 are already normal form so exclude them when normalize data
train_data_norm<-train_data[,-7:-9]
test_data_norm<-test_data[,-7:-9]
#Normalize data
train_data_norm<-as.data.frame(lapply(train_data_norm[ ,-8], data_norm))
test_data_norm<-as.data.frame(lapply(test_data_norm[ ,-8], data_norm))
#Add column 7 to 9 back
train_data_norm<-cbind(train_data_norm, train_data[,7:9])
test_data_norm<-cbind(test_data_norm, test_data[,7:9])


#Build knn model

library(class)

#When k=1
pred_data1<-knn(train_data_norm, test_data_norm, train_data[,11], k=1)

#Confusion matrix
install.packages('caret')
library(caret)
con_matr1<-confusionMatrix(data=as.factor(pred_data1),reference = as.factor(test_data[,11]))


#When k=3
pred_data2<-knn(train_data_norm, test_data_norm, train_data[,11], k=3)
confusionMatrix(data=as.factor(pred_data2),reference = as.factor(test_data[,11]))

#Find the best k-value by using accuracy as reference
n<-1
k_opt=1
for(n in 1:28){
  k_mod<-knn(train_data_norm, test_data_norm, train_data[,11], k=n)
  k_opt[n]<-confusionMatrix(data=as.factor(k_mod),
                            reference = as.factor(test_data[,11]))$overall[["Accuracy"]]
  k=n
  cat(k, '=', k_opt[n], '\n')
  
}

#Draw the plot of accuracy and k-value
plot(k_opt, type='b', xlab='K-value', ylab='Accuracy level')


### The second technique 
library(GGally)
library(caret)
library(corrplot)
library(psych)
library(rpart)
library(dplyr)
library(e1071)


#### Importing data
df <- read.table("../diabetic_data.csv", sep = ",", header = T, na.strings = "?")
summary(df)
head(df)

#### Excluding irrelevant columns
data <- select(df,  -encounter_id, -patient_nbr, -weight,-(25:41),-(43:47))
head(data)
summary(data)
dim(data)

#### CORRELATION PLOT
numeric_data <- select_if(data,is.numeric)
numeric_data
dim(numeric_data)
c <- cor(numeric_data, use= "pairwise.complete.obs")
corrplot(c)

#### Cleaning Data
data$race[is.na(data$race)] <- "Other"


#### Exploratory Data Analysis
plot(data$gender, main = "Distribution of Gender") 
plot(data$age, main = "Distribution of Age")
plot(data$A1Cresult, main = "A1C") 
plot(data$readmitted, main = "# of Readmissions") 
plot(data$admission_source, main = "Source of Admissions") 


#### Altering categorical variables
num_data <- data
num_data$diag_1 <- as.numeric(levels(num_data$diag_1)[num_data$diag_1])
num_data$diag_2 <- as.numeric(levels(num_data$diag_2)[num_data$diag_2])
num_data$diag_3 <- as.numeric(levels(num_data$diag_3)[num_data$diag_3])

#### PCA Selection
numeric_data<-scale(numeric_data)
pcaObj <- princomp(numeric_data, cor = TRUE, scores = TRUE, covmat = NULL)
summary(pcaObj)
print(pcaObj)
names(pcaObj)
plot(pcaObj)
pcaObj$loadings
final_data <- as.data.frame(pcaObj$scores)
final_data

#### Converting the dataset into training data and validation data

inTrain <- createDataPartition(y = num_data$readmitted, p = .67,list = FALSE)
train <- num_data[ inTrain,]
test <- num_data[-inTrain,]
dim(train)
dim(test)

###export train data and test data
write.csv(train, '/Users/siddharthsaxena/Desktop/logitical_regression_model_train_data.csv', row.names = FALSE)
write.csv(test, '/Users/siddharthsaxena/Desktop/logitical_regression_model_test_data.csv', row.names = FALSE)


train_nonbinary<-train
test_nonbinary<-test
train$readmitted<-ifelse(train$readmitted ==train$readmitted[1],0,1)
test$readmitted<-ifelse(test$readmitted ==test$readmitted[1],0,1)
test

#Adding dimension which are most significant and then doing logistic regression

logistic_model <- glm(readmitted ~ race+age+time_in_hospital+
                    num_lab_procedures+number_outpatient+
                    number_emergency+number_inpatient+number_diagnoses+
                    insulin+diabetesMed+A1Cresult,
                  data=train, family=binomial)

summary(logistic_model)

pred_logit <- predict(logistic_model,test, type = "response")
pred_logit
pred_logit <- ifelse(pred_logit > 0.5, 1, 0)
pred_logit
result<-as.data.frame(table(pred_logit,test$readmitted))
result
CorrectlyPredicted <- result[1,3]+result[4,3]
accuracy <-CorrectlyPredicted/nrow(test)
accuracy

senstivity_result<-result[4,3]/(result[2,3]+result[4,3])
senstivity_result
specificity_result<-result[1,3]/(result[3,3]+result[1,3])
specificity_result

library(pROC)
test_prob = predict(logistic_model, newdata = test, type = "response")
test_roc = roc(test$readmitted ~ pred_logit, plot = TRUE, print.auc = TRUE)
