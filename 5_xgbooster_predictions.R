library(xgboost)

for (i in 1:30){
  
  train <- read.csv("./clean_Dataset/train.csv", stringsAsFactors = FALSE)
  labels <- read.csv("./raw_Dataset/target.csv", stringsAsFactors = FALSE)
  
  target <- rep(0,nrow(train))
  target[labels$status_group == "non functional"] <- 1
  target[labels$status_group == "functional needs repair"] <- 2
  
  train <- train[,-1] 
  predictors <- data.matrix(train)
  rm(train)
  
  bst <- xgboost(data = predictors,
                 label = target,
                 num_class = 3,
                 max_depth = 17,
                 eta = 0.02,
                 nthread = 12,
                 subsample = 0.8,
                 colsample_bytree = 0.5,
                 min_child_weight = 1,
                 nrounds = 600, 
                 objective = "multi:softprob",
                 maximize = FALSE)
  
  rm(predictors)
  
  test <- read.csv("./clean_Dataset/test.csv", stringsAsFactors = FALSE)
  test_id <- test[,1:2]
  test <- test[,-1] 
  
  predictors_test <- data.matrix(test)
  
  rm(test)
  
  predictions <- predict(bst, predictors_test)
  
  results <- data.frame(test_id$id,
                        predictions[(1:length(predictions)) %% 3 == 1],
                        predictions[(1:length(predictions)) %% 3 == 2],
                        predictions[(1:length(predictions)) %% 3 == 0])
  
  colnames(results)[2:4] <- c("functional", "non functional","functional needs repair")
  
  write.csv(results, file = paste0("./predictions/prediction_", i, ".csv"), row.names = FALSE)
  rm(predictors_test)
}