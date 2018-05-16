
# Training an xgboost model on previously engineered features
# This model stacks the logisitic regression score with the hand-crafted features
# It is trained and evaluated on hold-out datasets (train2, and dev2)

project_path <- 'F:/AdTracking/'
plotpath <- paste0(project_path, "graphs/")
mungepath <- paste0(project_path, "munge/")
configpath <- paste0(project_path, "config/")
diagnostic <- paste0(project_path, "diagnostic/")
modelpath <- paste0(project_path, "model/")
output <- paste0(project_path, "output/")

###################################################################################################
######################################  LOADING LIBRARIES  ########################################
###################################################################################################

# install_version("xgboost", version = "0.4-4", repos = "http://cran.us.r-project.org")

library(xgboost)
library(data.table)
library(ggplot2)
library(dplyr)
library(plotly)


###################################################################################################
###########################################  READING DATA  ########################################
###################################################################################################

feature.names <- fread(paste0(mungepath,'feature.names.csv'), header = T, sep = ',', data.table = FALSE)
feature.names <- feature.names$feature


train2 <- fread(paste0(mungepath,'model_data/20180504/rf_lr_lasso_inter2_noip/train_features/train_features.csv'),
                header = T, sep = ',', data.table = FALSE)
train2 <- subset(train2, select=c('id', 'is_attributed', feature.names))


dev2 <- fread(paste0(mungepath,'model_data/20180504/rf_lr_lasso_inter2_noip/validation_features/validation_features.csv'),
              header = T, sep = ',', data.table = FALSE)
dev2 <- subset(dev2, select=c('id', 'is_attributed', feature.names))


test <- fread(paste0(mungepath,'model_data/20180504/rf_lr_lasso_inter2_noip/test_features/test_features.csv'),
              header = T, sep = ',', data.table = FALSE)
test <- subset(test, select=c('click_id', 'id', feature.names))


###################################################################################################
#######################################  DATA PREPROCESSING  ######################################
###################################################################################################

# convert integer variables to numeric
for(f in feature.names){
  if(class(train2[[f]])=="integer"){
    train2[[f]] <- as.numeric(train2[[f]])
    dev2[[f]] <- as.numeric(dev2[[f]])
    test[[f]] <- as.numeric(test[[f]])
  }
}

###################################################################################################
#######################################  CORRELATION ANALYSIS  ####################################
###################################################################################################

require(corrplot)

M <- cor(train2[,feature.names])
corrplot(M)


###################################################################################################
#######################################  DISTRIBUTION ANALYSIS  ###################################
###################################################################################################

# study the differences in distrubtion for each variable between dev and test set

if(!file.exists(paste0(plotpath, "dev2_test_comparison"))){
  dir.create(paste0(plotpath, "dev2_test_comparison"), recursive=T)
}

for(f in c("ip", "app", "device", "os", "channel", "day", feature.names)){
  
  set.seed(2210)
  dev2_length <- nrow(dev2)
  test_length <- nrow(test)
  
  hist_data <- rbind(data.frame(value = dev2[[f]][sample(1:dev2_length, 1000000, replace=F)], set = "dev2"),
                     data.frame(value = test[[f]][sample(1:test_length, 1000000, replace=F)], set = "test"))
  
  png(paste0(plotpath, "dev2_test_comparison/", f, ".png"), width=1080, height=920)
  
  g <- ggplot(hist_data, aes(value, fill = set)) +
    geom_density(alpha = 0.2) +
    ggtitle(paste("Feature:", f))
  
  print(g)
  
  dev.off()
}


###################################################################################################
#######################################  XGBOOOST MODELING  #######################################
###################################################################################################

dtrain <- xgb.DMatrix(data=data.matrix(train2[,feature.names]), label=train2$is_attributed, missing=NA)
dval <- xgb.DMatrix(data=data.matrix(dev2[,feature.names]), label=dev2$is_attributed, missing=NA)
dtest <- xgb.DMatrix(data=data.matrix(test[,feature.names]), missing=NA)

set.seed(2210)
param <- list(objective           = 'binary:logistic',
              booster             = 'gbtree',
              eval_metric         = 'auc',
              eta                 = 0.023, # 0.06, #0.01,
              max_depth           = 6, #changed from default of 8
              subsample           = 0.96, # 0.7
              colsample_bytree    = 1, # 0.7
              min_child_weight    = 10,
              nthread             = 20)

# training the xgboost model
xgbCV <- xgb.cv(params                 = param,
                data                   = dtrain,
                nrounds                = 20000,
                verbose                = 1,
                early_stopping_rounds  = 100,
                nfold                  = 5,
                maximize               = T)


# saving the model
save(xgbCV, file=paste0(modelpath, "xgbCV_lr_lasso_inter2.RData"))
load(paste0(modelpath, "xgbCV_lr_lasso_inter2.RData"))

# Plot learning curves
learning_curves <- xgbCV$evaluation_log

plot_ly(learning_curves, x = ~iter) %>%
  add_trace(y = ~train_auc_mean, name = 'train', type = 'scatter', mode = 'lines') %>%
  add_trace(y = ~test_auc_mean, name = '5-fold CV', type = 'scatter', mode = 'lines') %>%
  layout(title = 'Learning curves',
         xaxis = list(title = 'Iteration', tickfont = list(size = 10), tickangle = 45),
         yaxis = list(title = 'AUC', tickfont = list(size = 10)))


# find the best iteration regarding the cv performance
test.auc <- xgbCV$evaluation_log[['test_auc_mean']]
best.iter <- which.max(test.auc)
max(test.auc)

# training the xgboost model
watchlist<-list(train=dtrain, test=dval)

xgbMod <- xgb.train(params                 = param, 
                    data                   = dtrain, 
                    nrounds                = best.iter, 
                    verbose                = 1,  #1
                    early_stopping_rounds  = 100,
                    watchlist              = watchlist,
                    maximize               = T)


# saving the model
save(xgbMod, file=paste0(modelpath, "xgb_lr_lasso_inter2.RData"))

# Compute feature importance matrix
importance_matrix <- xgb.importance(feature.names, model = xgbMod)
write.csv(importance_matrix,
          paste0(diagnostic, "variable_importance_xgb.csv"),
          row.names=F)

View(importance_matrix)

# Nice graph
require(Ckmeans.1d.dp)
xgb.plot.importance(importance_matrix[1:20,])

# Understand interactions between features
require(DiagrammeR)
xgb.plot.tree(feature_names = feature.names, model = xgbMod, n_first_tree = 2)


###################################################################################################
#############################################  SUBMISSION  ########################################
###################################################################################################

# generate predictions
testpred <- round(predict(xgbMod, dtest),6)
mean(testpred) # 0.02489141

# Predict on test set
submission <- data.frame(click_id = test$click_id,
                         is_attributed = testpred)

# submission file
write.csv(submission, paste0(output, 'submission.csv'), row.names=F)


# SUMMARY
# missing             = NA
# eta                 = 0.023
# max_depth           = 6, #changed from default of 8
# subsample           = 0.96, # 0.7
# colsample_bytree    = 1, # 0.7
# min_child_weight    = 10,
# nround = 20000
# tr score: 0.988940+0.000050
# cv score: 0.981479+0.000308
# val score: 0.982092
# LB score: 0.9750


