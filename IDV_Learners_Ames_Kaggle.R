library(tidyverse)
library(caret)
library(scales)
library(corrplot)
library(RColorBrewer)
library(pls)
library(car)
library(grid)
library(lattice)
library(e1071)
library(ranger)
library(extraTrees)
library(RRF)
library(Rborist)
library(Rcpp)
library(quantreg)
library(quantregForest)
library(doParallel)
library(rpart)
library(rpart.plot)


setwd("C:\\Users\\UserPC\\Documents\\Data Analyst Practice\\Ames_Kaggle\\Test_Run")
dataset <- read.csv("cleaned_housing.csv", header = TRUE)
dataset_train <- read.csv("train_housing.csv", header = TRUE)
dataset_test <- read.csv("test_housing.csv", header = TRUE)


#Count of houses sold according to prices
ggplot(data=dataset_train[!is.na(dataset_train$saleprice),], aes(x=saleprice)) +
  geom_histogram(color = "black", fill="goldenrod3", binwidth = 10000) +
  scale_x_continuous(breaks= seq(0, 800000, by=100000), labels = comma) +
  geom_vline(aes(xintercept = mean(saleprice), linetype = "Mean Sale Price"), color = 'red')

#mean sale price
mean(dataset$saleprice)


#Sales Prices according to Neighborhood against mean prices
ggplot(dataset_train, aes(x = neighborhood, y = saleprice)) + 
  geom_boxplot(fill = "#81AC9B") +
  theme(axis.text.x = element_text(angle = 90, size = 8), legend.position = "none") + 
  scale_y_continuous("Sale Price", labels = dollar) + 
  geom_hline(aes(yintercept=mean(saleprice)), colour='red', linetype='solid', lwd=1)
scale_x_discrete("Neighborhood")

#Age of homes that are being sold
ggplot(dataset_train, aes(x = (yr.sold - year.built), y=saleprice))+
  geom_bar(stat="identity", fill="light blue")+
  scale_x_continuous(breaks=seq(0,150, 10))+
  scale_y_continuous(labels = comma)




#correlation plot
numericVars <- which(sapply(dataset_train, is.numeric)) #index vector numeric variables
numericVarNames <- names(numericVars) #saving names vector for use later on
corr_train <- cor(dataset_train[,numericVars], use='pairwise.complete.obs')
cor_sorted <- as.matrix(sort(corr_train[,'saleprice'], decreasing = TRUE))
cor_sorted

CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x)>.5)))
cor_numVar <- corr_train[CorHigh, CorHigh]

corrplot.mixed(cor_numVar, tl.col="black", tl.pos = "lt", upper="shade")


trimmed_corr_train <- as.data.frame(apply(cor_sorted, 2, function(x) ifelse (abs(x) >=0.5,x,NA)))
na.omit(trimmed_corr_train)

#saleprice vs. overall quality

ggplot(data=dataset_train[!is.na(dataset_train$saleprice),], 
       aes(x=factor(overall.qual), y=saleprice)) +
  geom_boxplot(col='sienna4', fill='goldenrod3') + labs(x='Overall Quality') +
  scale_y_continuous(breaks= seq(0, 800000, by=50000), labels = comma)+
  ggtitle('Sale price vs. Overall Quality')

# Unit count vs. Overall Quality bar chart
ggplot(data=dataset_train[!is.na(dataset_train$saleprice),], 
       aes(x=factor(overall.qual))) +
  geom_bar(col='sienna4', fill='goldenrod3') + labs(x='Overall Quality') +
  ggtitle('Units sold vs. Overall Quality')

#Sale Price vs. Ground Living Area
ggplot(subset(dataset_train, gr.liv.area > 0 & gr.liv.area < 4000), 
       aes(gr.liv.area, saleprice)) + 
  geom_point(color = "mediumvioletred") + geom_smooth(method = "loess") +
  annotate("text", x = 800, y = 5.5e+05, 
           label = paste('Cor =', 
                         round(cor(dataset_train$gr.liv.area, dataset_train$saleprice),5)), 
           size = 7, color = 'red') +
  scale_y_continuous(breaks= seq(0, 800000, by=50000), labels = comma)+
  xlab('Gr Liv Area (square feet)') + 
  ylab('Sale Price ($)') +
  ggtitle('Sale price vs. Gr Liv Area')


# specify actual values in test set for model predictions
test_y <- dataset_test[, "saleprice"]

# convert factors to integers in train set
i <- sapply(dataset_train, is.factor)

dataset_train[i] <- lapply(dataset_train[i], as.integer)
# convert factors to integers in test set
i <- sapply(dataset_test, is.factor)
dataset_test[i] <- lapply(dataset_test[i], as.integer)


#GLM model against all variables
all.model <- glm(saleprice ~ ., data = dataset_train)
# predict on test set
pred.all <- predict(all.model, dataset_test)
RMSE.all <- sqrt(mean((pred.all - test_y)^2))
RMSE.all

#RMSE Results table
Results <- data_frame(Method="Multivariate GLM - All variables", RMSE = round(RMSE.all,2))
Results

##Linear Model for variables with correlation above 0.5
Linear_Model_Above_0.5 <- glm(data = dataset_train, saleprice ~ 
                               overall.qual+
                               neighborhood +
                               gr.liv.area +
                               garage.cars +
                               garage.area +
                               total.bsmt.sf + 
                               x1st.flr.sf +
                               year.built +
                               full.bath +
                               year.remod.add +
                               yrs.since.remod +
                               house.age)



LM_pred <- predict(Linear_Model_Above_0.5, dataset_test)
df_LM_pred <- data.frame(LM_pred, dataset_test$saleprice)
saleprice_MSE.1 <- mean((df_LM_pred$LM_pred - df_LM_pred$dataset_test.saleprice)^2, na.rm=TRUE)
RMSE_LM_pred <- sqrt(saleprice_MSE.1)

RMSE_LM_pred

Results <- bind_rows(Results,
                     data_frame(Method="Multivariate GLM - >0.5 corr variables",  
                                RMSE = RMSE_LM_pred))
Results

#Principal Component Regression
pcr_train <- read.csv("train_housing.csv", header = TRUE)
pcr_train_num <- pcr_train
pcr_test <- read.csv("test_housing.csv", header = TRUE)
# drop factors
for(i in colnames(pcr_train)){
  if(is.factor( pcr_train[[i]] )){
    pcr_train_num[[i]] <- NULL
  }
}


pcr_saleprice <- pcr(saleprice ~ ., data=pcr_train_num, scale = TRUE, validation = "CV")
plot(pcr_saleprice,"validation")
axis(1, at=seq(0, 40, 2))

summary(pcr_saleprice)


# predict RMSE pcr1
pcr_predict_1 <- predict(pcr_saleprice, pcr_test, ncomp=1)
pcr_predict_1 <- data.frame(pcr_predict_1, pcr_test$saleprice)
saleprice_MSE_1 <- mean((pcr_predict_1$saleprice.1.comps - pcr_predict_1$pcr_test.saleprice)^2)
pcr_RMSE_1 <- sqrt(saleprice_MSE_1)
pcr_RMSE_1

Results <- bind_rows(Results,
                     data_frame(Method="PCR - 1 component",  
                                RMSE = pcr_RMSE_1))

Results

# predict RMSE pcr2
pcr_predict_2 <- predict(pcr_saleprice, pcr_test, ncomp=7)
pcr_predict_2 <- data.frame(pcr_predict_2, pcr_test$saleprice)
saleprice_MSE_2 <- mean((pcr_predict_2$saleprice.7.comps - pcr_predict_2$pcr_test.saleprice)^2)
pcr_RMSE_2 <- sqrt(saleprice_MSE_2)
pcr_RMSE_2

Results <- bind_rows(Results,
                     data_frame(Method="PCR - 7 component",  
                                RMSE = pcr_RMSE_2))

Results

# predict RMSE pcr3
pcr_predict_3 <- predict(pcr_saleprice, pcr_test, ncomp=24)
pcr_predict_3 <- data.frame(pcr_predict_3, pcr_test$saleprice)
saleprice_MSE_3 <- mean((pcr_predict_3$saleprice.24.comps - pcr_predict_3$pcr_test.saleprice)^2)
pcr_RMSE_3 <- sqrt(saleprice_MSE_3)
pcr_RMSE_3


Results <- bind_rows(Results,
                     data_frame(Method="PCR - 24 component",  
                                RMSE = pcr_RMSE_3))
# new df
output_df <- data.frame(pcr_predict_1$pcr_test.saleprice, pcr_predict_1$saleprice.1.comps, pcr_predict_2$saleprice.7.comps, pcr_predict_3$saleprice.24.comps)
# rename df
output_df <- dplyr::rename(output_df, test_saleprice=pcr_predict_1.pcr_test.saleprice, Prediction.1.Component=pcr_predict_1.saleprice.1.comps, Prediction.7.Component=pcr_predict_2.saleprice.7.comps, Prediction.24.Component=pcr_predict_3.saleprice.24.comps)



Results 


###Random Forest

# make a dataframe of all numeric features
num_features = names(which(sapply(dataset_train, is.numeric)))
df_numeric = dataset_train[num_features]
# make a dataframe of all categorical features
cat_features = names(which(sapply(dataset_train, is.factor)))
# convert all categorical features to numeric variables for modeling
dataset_train[cat_features] <- sapply(dataset_train[cat_features], as.integer)
dataset_test[cat_features] <- sapply(dataset_test[cat_features], as.integer)
# split test into x and y
test_x <- dataset_test %>% dplyr::select(-saleprice)
test_y <- dataset_test %>% dplyr::select(saleprice)

# set seed for reproducibility
set.seed(256)
##myFolds <- createFolds(dataset_train, k = 5)
control <- trainControl(method = "repeatedcv", repeats = 3, number = 10)
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

#Conditional Inference Random Forest (cforest)
set.seed(123)

cforest_model <- train(saleprice ~ ., 
        method = "cforest", 
        data = dataset_train,
        trControl = control)

pred_cforest <- predict(cforest_model, dataset_test)
df_pred_cforest <- data.frame(pred_cforest, dataset_test$saleprice)
saleprice_cforest <- mean((df_pred_cforest$pred_cforest - df_pred_cforest$dataset_test.saleprice)^2)
qrf_RMSE <- sqrt(saleprice_cforest)
Results <- bind_rows(Results,
          data_frame(Method="cforest",  
                     RMSE = cforest_RMSE))

#QRF
set.seed(123)

qrf_model <- train(saleprice ~ ., 
                   method = "qrf", 
                   data = dataset_train,
                   trControl = control)

pred_qrf <- predict(qrf_model, dataset_test)
df_pred_qrf <- data.frame(pred_qrf, dataset_test$saleprice)
saleprice_qrf <- mean((df_pred_qrf$pred_qrf - df_pred_qrf$dataset_test.saleprice)^2)
qrf_RMSE <- sqrt(saleprice_qrf)
Results <- bind_rows(Results,
                     data_frame(Method="Quantile Random Forest (QRF)",  
                                RMSE = qrf_RMSE))


#ranger
set.seed(123)

ranger_model <- train(saleprice ~., 
                   method = "ranger", 
                   data = dataset_train, 
                   trControl = control)

pred_ranger <- predict(ranger_model, dataset_test)
df_pred_ranger <- data.frame(pred_ranger, dataset_test$saleprice)
saleprice_ranger <- mean((df_pred_ranger$pred_ranger - df_pred_ranger$dataset_test.saleprice)^2)
ranger_RMSE <- sqrt(saleprice_ranger)
Results <- bind_rows(Results,
          data_frame(Method="Random Forest (Ranger)",  
                     RMSE = ranger_RMSE))

#Rborist
set.seed(123)

Rborist_model <- train(saleprice ~., 
                      method = "Rborist", 
                      data = dataset_train, 
                      trControl = control)

pred_Rborist <- predict(Rborist_model, dataset_test)
df_pred_Rborist <- data.frame(pred_Rborist, dataset_test$saleprice)
saleprice_Rborist <- mean((df_pred_Rborist$pred_Rborist - df_pred_Rborist$dataset_test.saleprice)^2)
Rborist_RMSE <- sqrt(saleprice_Rborist)
Results <- bind_rows(Results,
          data_frame(Method="Random Forest (Rborist)",  
                     RMSE = Rborist_RMSE))

#RF
set.seed(123)

RF_model <- train(saleprice ~ ., 
                  method = "rf", 
                  data = dataset_train,
                  trControl = control)

pred_RF <- predict(RF_model, dataset_test)
df_pred_RF <- data.frame(pred_RF, dataset_test$saleprice)
saleprice_RF <- mean((df_pred_RF$pred_RF - df_pred_RF$dataset_test.saleprice)^2)
RF_RMSE <- sqrt(saleprice_RF)
Results <- bind_rows(Results,
          data_frame(Method="Random Forest (RF)",  
                     RMSE = RF_RMSE))


#parRF
set.seed(123)

parRF_model <- train(saleprice ~ ., 
                  method = "parRF", 
                  data = dataset_train, 
                  trControl = control)

pred_parRF <- predict(parRF_model, dataset_test)
df_pred_parRF <- data.frame(pred_parRF, dataset_test$saleprice)
saleprice_parRF <- mean((df_pred_parRF$pred_parRF - df_pred_parRF$dataset_test.saleprice)^2)
parRF_RMSE <- sqrt(saleprice_parRF)
Results <- bind_rows(Results,
          data_frame(Method="Parallel Random Forest (parRF)",  
                     RMSE = parRF_RMSE))

stopCluster(cl)

Results
