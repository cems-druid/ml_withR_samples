#Codes are from the book of Machine Learning Mastery with R, Jason Brownlee.
#9 Pre-Processing 
#9.3 Scale data: Scale transform calculates the s.d. for each an attribute and divides each value by that standard deviation. 
library(caret)
data(iris)
summary(iris[,1:4])
preprocess_parameters <- preProcess(iris[1:4], method = c("scale"))
print(preprocess_parameters)
transformed <- predict(preprocess_parameters, iris[,1:4])
summary(transformed)

#9.4 Center data: Center transforms calculates the mean for an attribute and substracts it from the each value.
preprocess_parameters_center <- preProcess(iris[,1:4], method = c("center"))
print(preprocess_parameters_center)
transformed_center <- predict(preprocess_parameters_center, iris[,1:4])
summary(transformed_center)

#9.5 Standardize data: Combine scale and center transforms to standardize data. Attributes will have a mean value of 0 and standard deviation of 1.

preprocess_sc <- preProcess(iris[,1:4], method = c("scale", "center"))
print(preprocess_sc)
transformed_sc <- predict(preprocess_sc, iris[,1:4])
summary(transformed_sc)

#9.5 Normalize data: Values are put into range of 0 and 1.
preprocess_norm <- preProcess(iris[,1:4], method = c("range"))
print(preprocess_norm)
transformed_norm <- predict(preprocess_norm, iris[,1:4])
summary(transformed_norm)

#9.6 Box-Cox transform: Used for making skewed gaussian plots, less skewed. All values have to be positive.
library(mlbench)
data("PimaIndiansDiabetes")
summary(PimaIndiansDiabetes[,7:8])
preprocess_boxcox <- preProcess(PimaIndiansDiabetes[,7:8], method= c("BoxCox"))
print(preprocess_boxcox)
transformed_boxcox <- predict(preprocess_boxcox, PimaIndiansDiabetes[,7:8])
summary(transformed_boxcox)

#9.6 Yeo-Johnson transform: same operations as box-cox but can take negative and zero values.

summary(PimaIndiansDiabetes[,7:8])
preprocess_yeojohn <- preProcess(PimaIndiansDiabetes[,7:8], method=c("YeoJohnson"))
print(preprocess_yeojohn)
transformed_yeojohn <- predict(preprocess_yeojohn, PimaIndiansDiabetes[,7:8])
summary(transformed_yeojohn)

#9.9. Principal Component Analysis (PCA) transform: It is a technique from multrivariate statictics and linear algebra that returns 
# only the principal components. The transform keeps those components above the variance threshold (default=0.95) or the number of 
#components can be specifed by pcaComp. The results is attributes that are uncorrelateed, useful for algorithms like linear and generalized
#linear regression.

data(iris)
summary(iris)
preprocess_pca <- preProcess(iris, method=c("center", "scale", "pca"))
print(preprocess_pca)
transformed_pca <- predict(preprocess_pca, iris)
summary(transformed_pca)

#9.10 Independent Component Analysis (ICA) transform: Transform the data to independent components and returns the components that are independent.
#This transform might be useful with algorithms such as Naive-Bayes. 

data("PimaIndiansDiabetes")
summary(PimaIndiansDiabetes[,1:8])
preprocess_ica <- preProcess(PimaIndiansDiabetes[,1:8], method=c("center", "scale", "ica"), n.comp=5)
print(preprocess_ica)
transformed_ica <- predict(preprocess_ica, PimaIndiansDiabetes[,1:8])
summary(transformed_ica)

#10 Resampling to Estimating Accuracy
#10.2 Data Split: Partitioning the data into training and testing datasets. 
library(klaR)
data(iris)
train_index_iris <- createDataPartition(iris$Species, p=0.8, list=FALSE)
iris_train <- iris[train_index_iris,]
iris_test <- iris[-train_index_iris,]
#naive bayes modelling
iris_fit_nb <- NaiveBayes(Species~., data=iris_train)
iris_predict_nb <- predict(iris_fit_nb, iris_test[,1:4])
confusionMatrix(iris_predict_nb$class, iris_test$Species)

#10.3 Bootstrap: Bootstrap resampling involves taking random samples from the dataset (with re-selection) against which to evaluate the model.
#In aggragate the results provide an indication of the variance of the models performance. Typically, large number of resampling iterations are performed.
trainControl_bs <- trainControl(method = "boot", number = 100)
fit_bs <- train(Species~., data=iris, trControl=trainControl_bs, method ="nb")
print(fit_bs)

#10.4 k-fold Cross Validation: K-fold cross validation method involves splitting the dataset into k-subsets, which each subset is held
#out while the model is trained on other subsets. This process is completed until accuracy is determine for each instance in the dataset 
#and an overall accuracy estimate is provided. It is a robust method for estimating accuracy and the size of k can tune the amount of bias
#in the estimate, popularly 5 and 10 are used.
trainControl_kf <- trainControl(method = "cv", number = 10)
fit_kf <- train(Species~., data=iris, trControl=trainControl_kf, method ="nb")
print(fit_kf)

#10.5 Repeated k-fold Cross Validation: Same as above but only repeats itself number of times and the mean of the accuracy is taken as final.
trainControl_rkf <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
fit_rkf <- train(Species~., data=iris, trControl = trainControl_rkf, method = "nb")
print(fit_rkf)

#10.6 Leave One Out Cross Validation (LOOCV): a data instance is left out and a model constructed on all the other data instances in the training set. 
trainControl_loocv <- trainControl(method = "LOOCV")
fit_loocv <- train(Species~., data=iris, trControl = trainControl_loocv, method = "nb")
print(fit_loocv)


#11 Model Evaluation Metrics:
#Classification -> Accuracy, Kappa, Logarithmic Loss, Binary Classification -> Area Under ROC Curve, sensivity, specificity
#Regression -> RMSE and R^2

#11.2 Accuracy and Kappa: Defualt metrics for binary and multiclass classification. Accuracy is good with binary classification. 
#Kappa is like accuracy but it is normalized at the baseline of random chance on your dataset, good in sets with imbalances on target data.
data("PimaIndiansDiabetes")
trainControl_ak <- trainControl(method="cv", number=5)
set.seed(7)
fit_ak <- train(diabetes~., data=PimaIndiansDiabetes, method = "glm", metric = "Accuracy", trControl= trainControl_ak)
print(fit_ak)

#11.3 RMSE and R^2: default metrics to evaluate regression dataset (in caret). RMSE = Root Mean Squared Error, is the average of the deviation 
#of predictions from the observations. It is good with general goodness of an algorithm. 
#R^2 or the coefficient of determination provides a goodness of-fit measure for the predictions to the observations. This is a value
#between 0 and 1 for not-fit and perfect.
data("longley")
trainControl_rmser <- trainControl(method = "cv", number = 5)
set.seed(9)
fit_rmser <- train(Employed~., data=longley, method = "lm", metric = "RMSE", trControl = trainControl_rmser)
print(fit_rmser)

#11.4 Area under ROC Curve (AUC): ROC metrics are only suitable for binary classification problems, the AUC represents a models ability to
#discriminate between positive and negative classes. An area of 1.0 represents a model that predicts perfectly. AN are of 0.5 represents a model as good as random.
#Sensitiviy = Recall (True Positive), Specificity = True Negative. A binary classification problem is a trade of between sensivity and specificity.
data("PimaIndiansDiabetes")
trainControl_auc <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
set.seed(10)
fit_auc <- train(diabetes~., data=PimaIndiansDiabetes, method = "glm", metric = "ROC", trControl = trainControl_auc)
print(fit_auc)

#11.5 Logarithmic Loss: It evaluates the probabilities estimated by the algorithm.
data("iris")
trainControl_ll <- trainControl(method = "cv", number=5, classProbs = TRUE, summaryFunction = mnLogLoss)
set.seed(11)
fit_ll <- train(Species~., data=iris, method = "rpart", metric="logLoss", trControl=trainControl_ll)
print(fit_ll)

#12 Spot-Check Algorithms: 
#Which algorithm/model is good for dataset? We cannot know the best algorithm before we trial and error. Experience, trial and error. You can improve
#the results of candidate algorithms by either tuning the algorithm parameters or by combining the predictions of multiple models using ensemble methods.
#These algorithm categories should be checked: mixture of algorithm representations (e.g instance-based methods and trees), mixture of learning algorithms
#(e.g. different algorithms for learning the same type of representation), mixture of modeling types (e.g. linear and non-linear functions or parametric and
#non-parametric). 

#12.3 Linear Algorithms: So the algorithms are presented in two groups; linear algorithms: simpler methods that have strong bias but are fast to train and 
#non-linear algorithms: more complex methods and have large variance but are more accurate.
#12.3.1: Linear Regression
#lm()
data("BostonHousing")
fit_lr <- lm(medv~., data = BostonHousing)
print(fit_lr)
prediction_lr <- predict(fit_lr, BostonHousing)
mse_lr <- mean((BostonHousing$medv-prediction_lr)^2)
print(mse_lr)
#caret()
set.seed(13)
trainControl_lr <- trainControl(method = "cv", number=5)
fit_lr.lm <- train(medv~., data=BostonHousing, method="lm", metric="RMSE", preProc=c("center", "scale"), trControl=trainControl_lr)
print(fit_lr.lm)

#12.3.2: Logistic Regression
data("PimaIndiansDiabetes")
fit_logr <- glm(diabetes~., data=PimaIndiansDiabetes, family=binomial(link='logit'))
print(fit_logr)
prob_logr <- predict(fit_logr, PimaIndiansDiabetes[,1:8], type='response')
pred_logr <- ifelse(prob_logr > 0.5, 'pos', 'neg')
table(pred_logr, PimaIndiansDiabetes$diabetes)

#with caret()  --- does not work idk why it says wrong model type for classification
set.seed(14) 
trainControl_logr <- trainControl(method="cv", number = 5)
fit_logr.glm <- train(diabetes~., data=PimaIndiansDiabetes, method = "lm", metric="Accuracy", preProcess=c("center", "scale"), trControl = trainControl_logr)
print(fit_logr.glm)

#12.3.3: Linear Discriminant Analysis:
library(MASS)
fit_lda <- lda(diabetes~., data=PimaIndiansDiabetes)
print(fit_lda)
pred_lda <- predict(fit_lda, PimaIndiansDiabetes[,1:8])$class
table(pred_lda, PimaIndiansDiabetes$diabetes)
#caret()
set.seed(15)
trainControl_lda <- trainControl(method="cv", number = 5)
fit_lda.lda <- train(diabetes~., data=PimaIndiansDiabetes, method = "lda", metric = "Accuracy", preProcess=c("center","scale"), trControl= trainControl_lda)
print(fit_lda.lda)

#12.3.4 Regularized Regression: 
#classification
library(glmnet)
x_rr <- as.matrix(PimaIndiansDiabetes[,1:8])
y_rr <- as.matrix(PimaIndiansDiabetes[,9])
fit_rr <- glmnet(x_rr, y_rr, family = "binomial", alpha = 0.5, lambda = 0.001)
print(fit_rr)
pred_rr <- predict(fit_rr, x_rr, type = "class") 
table(pred_rr, PimaIndiansDiabetes$diabetes)
#classification in caret()
set.seed(15)
trainControl_rr1 <- trainControl(method = "cv", number=5)
fit_rr1.glmnet <- train(diabetes~., data=PimaIndiansDiabetes, method="glmnet", metric = "Accuracy", preProcess=c("center","scale"), trControl = trainControl_rr1)
print(fit_rr1.glmnet)
#regression:
data("BostonHousing")
BostonHousing$chas <- as.numeric(as.character(BostonHousing$chas))
x_rr2 <- as.matrix(BostonHousing[,1:13])
y_rr2 <- as.matrix(BostonHousing[,14])
fit_rr2 <- glmnet(x_rr2, y_rr2, family = "gaussian", alpha = 0.5, lambda = 0.001)
print(fit_rr2)
pred_rr2 <- predict(fit_rr2, x_rr2, type="link")
mse_rr2 <- mean((y_rr2-pred_rr2)^2)
print(mse_rr2)
#regression in caret()
set.seed(16)
trainControl_rr3 <- trainControl(method="cv", number=5)
fit_rr3.glmnet <- train(medv~., data=BostonHousing, method="glmnet", metric = "RMSE", preProcess=c("center", "scale"), trControl = trainControl_rr3)
print(fit_rr3.glmnet)

#12.4 Non-Linear Algorithms: Less assumptions on models, consequently higher variance and often higher accuracy. More time and memory consumption.
#12.4.1 k-Nearest Neighbors: does not create a model just makes predictions from set. (caret())
#classification
data("PimaIndiansDiabetes")
fit_knn <- knn3(diabetes~., data=PimaIndiansDiabetes, k =3)
print(fit_knn)
pred_knn <- predict(fit_knn, PimaIndiansDiabetes[,1:8], type="class")
table(pred_knn, PimaIndiansDiabetes$diabetes)
#caret()
set.seed(16)
trainControl_knn1 <- trainControl(method="cv", number=5)
fit_knn1.knn <- train(diabetes~., data=PimaIndiansDiabetes, method = "knn", metric = "Accuracy", preProcess = c("center","scale"), trControl = trainControl_knn1 ) 
print(fit_knn1.knn)

#regression
data("BostonHousing")
BostonHousing$chas <- as.numeric(as.character(BostonHousing$chas))
x_knn <- as.matrix(BostonHousing[,1:13])
y_knn <- as.matrix(BostonHousing[,14])
fit_knn2 <- knnreg(x_knn, y_knn, k=3)
print(fit_knn2)
pred_knn2 <- predict(fit_knn2, x_knn)
mse_knn2 <- mean((BostonHousing$medv-pred_knn2)^2)
print(mse_knn2)
#caret()
set.seed(21)
trainControl_knn3 <- trainControl(method="cv",number=5)
fit_knn3.knn <- train(medv~., data=BostonHousing, method = "knn", metric = "RMSE", preProcess=c("center", "scale"), trControl = trainControl_knn3 )
print(fit_knn3.knn)

#12.4.2 Naive Bayes:
library(e1071)
fit_nb <- naiveBayes(diabetes~., data= PimaIndiansDiabetes)
print(fit_nb)
pred_nb <- predict(fit_nb, PimaIndiansDiabetes[,1:8])
table(pred_nb, PimaIndiansDiabetes$diabetes)
#with caret()
set.seed(22)
trainControl_nb <- trainControl(method="cv", number=5)
fit_nb.nb <- train(diabetes~., data=PimaIndiansDiabetes, method = "nb", metric = "Accuracy", trControl = trainControl_nb ) 
print(fit_nb.nb)

#12.4.3 Support Vector Machine (SVM)
#classification
library(kernlab)
fit_svm <- ksvm(diabetes~., data = PimaIndiansDiabetes, kernel = "rbfdot")
print(fit_svm)
pred_svm <- predict(fit_svm, PimaIndiansDiabetes[,1:8], type = "response")
table(pred_svm, PimaIndiansDiabetes$diabetes)
#caret()
set.seed(23)
trainControl_svm <- trainControl(method="cv", number = 5)
fit_svm1.svmRadial <- train(diabetes~., data=PimaIndiansDiabetes, method = "svmRadial",metric="Accuracy", trControl = trainControl_svm)
print(fit_svm1.svmRadial)
#regression
fit_svm2 <- ksvm(medv~., BostonHousing, kernel = "rbfdot")
print(fit_svm2)
pred_svm2 <- predict(fit_svm2, BostonHousing)
mse_svm2 <- mean((BostonHousing$medv-pred_svm2)^2)
print(mse_svm2)
#with caret()
set.seed(24)
trainControl_svm3 <- trainControl(method = "cv", number = 5)
fit_svm3.svmRadial <- train(medv~., data=BostonHousing, method="svmRadial", metric = "RMSE", trControl = trainControl_svm3 ) 
print(fit_svm3.svmRadial)

#12.4.4 Classification and Regression Trees
#classification
library(rpart)
fit_tree1 <- rpart(diabetes~., data=PimaIndiansDiabetes)
print(fit_tree1)
pred_tree1 <- predict(fit_tree1, PimaIndiansDiabetes[,1:8], type = "class")
table(pred_tree1, PimaIndiansDiabetes$diabetes)
#with caret()
set.seed(55)
trainControl_tree2 <- trainControl(method = "cv", number = 5)
fit_tree2.rpart <- train(diabetes~., data = PimaIndiansDiabetes, method = "rpart", metric = "Accuracy", trControl = trainControl_tree2)
print(fit_tree2.rpart)
#regression
fit_tree3 <- rpart(medv~., data = BostonHousing, control = rpart.control(minsplit = 5))
print(fit_tree3)
pred_tree3 <- predict(fit_tree3, BostonHousing[,1:13])
mse_tree3 <- mean((BostonHousing$medv-pred_tree3)^2)
print(mse_tree3)
#with caret()
set.seed(51)
trainControl_tree4 <- trainControl(method = "cv", number = 2)
fit_tree4.rpart <- train(medv~., data=BostonHousing, method = "rpart", metric = "RMSE", trControl = trainControl_tree4)
print(fit_tree4.rpart)

#13 Comparing ML Algorithm Performances
#This chapter involves three parts, preparing datasets, training models and comparing models.

library(mlbench)
library(caret)
data(PimaIndiansDiabetes)

#For training 5 models will be used with cross validation with 10 folds and 3 repeats including the 
#evaluation metric of kappa. Algorithms are; Classification and Regression Trees (CART), Linear Discriminant Analysis (LDA
#Support Vector Machine with Radial Basis Function (SVM), k-nearest neighbor (kNN), random forests (RF)
#mc -> model comparison
trainControl_mc  <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

#Classification and regression
set.seed(1627)
fit_mc.cart <- train(diabetes~., data = PimaIndiansDiabetes, method = "rpart", trControl = trainControl_mc)

#Linear Discriminant Analysis
set.seed(1627)
fit_mc.lda <- train(diabetes~., data = PimaIndiansDiabetes, method = "lda", trControl = trainControl_mc )

#Support Vector Machines
set.seed(1627)
fit_mc.svm <- train(diabetes~., data = PimaIndiansDiabetes, method = "svmRadial", trControl = trainControl_mc )

#K-Nearest Neighbor
set.seed(1627)
fit_mc.knn <- train(diabetes~., data = PimaIndiansDiabetes, method = "knn", trControl = trainControl_mc )

#Random Forest
set.seed(1627)
fit_mc.rf <- train(diabetes~., data = PimaIndiansDiabetes, method = "rf", trControl = trainControl_mc )

#Using resamples, this function checks if the same cv techniques are used.
results_mc <- resamples(list(CART=fit_mc.cart, LDA = fit_mc.lda, SVM = fit_mc.svm, KNN = fit_mc.knn, RF = fit_mc.rf))

summary(results_mc)


#plots; box and whisker plots
scales_mc <- list(x=list(relation="free"), y = list(relation="free"))
bwplot(results_mc, scales_mc)
#density plots
densityplot(results_mc, scales=scales_mc, pch="|" ) 
#dot plot
dotplot(results_mc, scales=scales_mc)
#parallel plot
parallelplot(results_mc)
#scatter plot
splom(results_mc)
#xyplot
xyplot(results_mc, models = c("LDA","SVM"))
#statistical significance test, p-values for pair-wise comparisons.
diffs_mc <- diff(results_mc)
summary(diffs_mc)

#14- Tuning Machine Learning Algorithms: