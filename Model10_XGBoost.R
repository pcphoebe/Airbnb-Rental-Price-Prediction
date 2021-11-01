### Model 10 - XGBoost ###

## @author Phoebe Chen
## @date 11/15/2020

##INITIAL EXPLORATION: read in data
analysisData = read.csv("analysisData.csv")
scoringData = read.csv("scoringData.csv")

#understand data
str(analysisData)
str(scoringData)
sum(is.na(analysisData))

##DATA CLEANING: combine data to clean together
scoringData$zipcode <- as.character(scoringData$zipcode)
library(dplyr)
combinedData <- bind_rows(analysisData, scoringData)

# clean up zipcode column
combinedData$zipcode <- substr(combinedData$zipcode, 1, 5) #only choose obtain first 5 digits of zipcode
combinedData$zipcode[nchar(combinedData$zipcode)<5] <- NA #set zipcodes that are less than 5 digits to NA
combinedData$zipcode <- as.factor(combinedData$zipcode) #convert to factor type
combinedData$zipcode <- forcats::fct_lump_n(combinedData$zipcode, 40) #lump together levels except for 40 most frequent levels
combinedData$zipcode[is.na(combinedData$zipcode)] <- "Other" #set NA zipcodes to "other"

# calculate median for all NA numeric variables
library(caret)
numeric_predictors <- which(colnames(combinedData) != "price" & sapply(combinedData, is.numeric))
imp_model_med <- preProcess(combinedData[ ,numeric_predictors], method = 'medianImpute')
combinedData[ ,numeric_predictors] <- predict(imp_model_med, newdata = combinedData[ ,numeric_predictors])

# Remove columns with little variance
zero_var_table <- nearZeroVar(combinedData, saveMetrics= TRUE) #obtain columns with 0 variance (hence little values or same values)
combinedData <- combinedData[, !zero_var_table$nzv] #remove these columns

# split back to train & test based on price as test set has no price column
train <- combinedData[!is.na(combinedData$price), ]
test <- combinedData[is.na(combinedData$price), ]

# remove character columns with more than 50 levels
train <- 
  train %>%
  mutate_if(is.character, factor) %>%
  select_if(~ nlevels(.) < 50)

# remove columns with > 90% NAs
train <- train[, which(colMeans(!is.na(train)) > 0.9)]

# find correlated variables
train_numeric_predictors <- which(sapply(train, is.numeric))
train_num <- train[, c(train_numeric_predictors)] 
library(corrplot)
corrplot(cor(train_num[,-16]),method = 'square',type = 'lower',diag = F) 

# double check highly correlated variables
cor(train$minimum_minimum_nights, train$minimum_nights) 
cor(train$minimum_maximum_nights, train$maximum_nights) 
cor(train$maximum_nights_avg_ntm, train$maximum_nights) 
cor(train$maximum_maximum_nights, train$maximum_nights) 
cor(train$calculated_host_listings_count, train$calculated_host_listings_count_entire_homes)
cor(train$availability_90, train$availability_60)  
cor(train$availability_30, train$availability_60) 
colnames(train) #find indexes
train <- train[ ,-c(1, 23, 25, 26, 28, 45, 29, 30)] #remove id & highly correlated columns

# ensure structure of train and that there are no NA values
str(train)
table(is.na(train))

## MODEL: XGBoost
library(vtreat)
trt = designTreatmentsZ(dframe = train, varlist = names(train)[-15])
#all columns other than price (column 16)
newvars = trt$scoreFrame[trt$scoreFrame$code %in% c('clean', 'lev'), 'varName']
#recode categorical attributes of training data as binary dummy variables
#apply these binary variables to train & test data
train_input = prepare(treatmentplan = trt,
                      dframe = train,
                      varRestriction = newvars)
test_input = prepare(treatmentplan = trt,
                     dframe = test,
                     varRestriction = newvars)

library(xgboost)
set.seed(617)
tune_nrounds = xgb.cv(data = as.matrix(train_input), 
                      label = train$price,
                      nrounds = 100,
                      nfold = 3,
                      verbose = 0)

xgboost = xgboost(data = as.matrix(train_input),
                  label = train$price,
                  nrounds = which.min(tune_nrounds$evaluation_log$test_rmse_mean),
                  verbose = 0)

#obtain rmse on train data
xgboost_train_pred = predict(xgboost, newdata = as.matrix(train_input))
rmse_xgboost = sqrt(mean((xgboost_train_pred - train$price)^2))
rmse_xgboost

#understanding relationship between predictors and price
importance_matrix <- xgb.importance(model = xgboost)
importance_matrix[order(importance_matrix[,2], decreasing = TRUE), ]
str(train)

#predict test data
xgboost_test_pred = predict(xgboost, newdata = as.matrix(test_input))
submissionFile = data.frame(id = test$id, price = xgboost_pred)
write.csv(submissionFile, "Model10 XGBoost.csv", row.names = FALSE)
