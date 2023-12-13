# GGG

# setwd("C:/Users/farri/OneDrive/Documents/PA/GGG")

# LIBRARIES=====================================================================

library(tidyverse)
library(vroom)
library(tidymodels)
library(discrim)
library(glmnet)
library(rpart)
library(ranger)
library(stacks)
library(recipes)
library(embed) 
library(naivebayes)
library(kknn)
library(themis) # for smote


# MISSING DATA==================================================================

test <- vroom("test.csv")
train <- vroom("train.csv")
sampleSubmission <- vroom("sample_submission.csv")

trainMissing <- vroom("trainWithMissingValues.csv")

step_impute_median()

Rec1 <- recipe(type ~ ., data = train) %>% 
  #step_impute_median() %>% 
  step_impute_linear(hair_length, impute_with=) %>% 
  step_impute_linear(bone_length, impute_with=) %>% 
  step_impute_linear(has_soul, impute_with=) %>% 
  step_impute_linear(rotting_flesh, impute_with=) %>% 
  step_impute_mode(color)

prepped_recipe <- prep(Rec1)
bake(prepped_recipe, new_data = train) #Make sure recipe work on train
trainNotMissing <- bake(prepped_recipe, new_data = trainMissing) #Make sure recipe works on test


rmse_vec(train[is.na(trainMissing)],trainNotMissing[is.na(trainMissing)])


## Create a workflow with model & recipe
Rec2 <- recipe(type ~ ., data = train)  %>%
  # step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric()) %>% 
  step_pca(all_numeric(), threshold=.90)

prep <- prep(Rec2)

bake(prep,new_data=train)


# step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
# # step_zv(all_predictors()) %>%
# #step_other(all_factor_predictors(), threshold = .01)  %>% # combines categorical values that occur <5% into an "other" value
# step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) #target encoding

## Set up K-fold CV


forest_model <- rand_forest(mtry = 1,
                            min_n=11,
                            trees=1000) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("classification")

forestwf <- workflow() %>%
  add_recipe(Rec2) %>%
  add_model(forest_model) %>%
  fit(data=train) 

predictions1 <- predict(forestwf,
                        new_data=test,
                        type="class") %>%  # "class" or "prob" (see doc) 
  bind_cols(., sampleSubmission) %>%
  select(id, .pred_class) %>%
  rename(type=.pred_class)


vroom_write(x=predictions1, file="./RFMN.csv", delim=",")

# Neural Networks===============================================================

nn_recipe <- recipe(type ~ ., data = train) %>%
  update_role(id, new_role="id") %>%
  step_dummy(color) %>% 
  step_range(all_numeric_predictors(), min=0, max=1) 


bake(prep(nn_recipe), new_data = train) #Make sure recipe work on train


nn_model <- mlp(hidden_units = 4, epochs = 50) %>%
  set_engine("nnet", verbose = 0) %>% 
  set_mode("classification")

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model) %>% 
  fit(data=train) 

predictions_s <- predict(nn_wf,
                         new_data=test,
                         type="class") %>%  # "class" or "prob" (see doc) 
  bind_cols(., sampleSubmission) %>%
  select(id, .pred_class) %>%
  rename(type=.pred_class)


vroom_write(x=predictions_s, file="./Prediction_f.csv", delim=",")

folds <- vfold_cv(train, v = 6, repeats=1)

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 20)),
                            levels=20)

tuned_nn <- nn_wf %>%
  tune_grid(resamples=folds,
            grid=nn_tuneGrid,
            metrics=metric_set(accuracy)) #Or leave metrics NULL

CV_results <- nn_wf %>%
  tune_grid(resamples=folds,
            grid=tuned_nn,
            metrics=metric_set(accuracy)) #Or leave metrics NULL

tuned_nn %>%
  select_best("accuracy")


tuned_nn %>% collect_metrics() %>%
  filter(.metric=="accuracy") %>%
  ggplot(aes(x=hidden_units, y=mean)) + geom_line()

## CV tune, finalize and predict here and save results
## This takes a few min (10 on my laptop) so run it on becker if you want



# Boosting =====================================================================

library(bonsai)
library(lightgbm)


boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

bart_model <- bart(trees=tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("classification")

## CV tune, finalize and predict here and save results

boo_wf <- workflow() %>%
  add_recipe(Rec2) %>%
  add_model(boost_model)

folds <- vfold_cv(train, v = 6, repeats=1)

boo_tuneGrid <- grid_regular(tree_depth(),
                            trees(),
                            learn_rate(),
                            levels=3)

CV_results <- boo_wf %>%
  tune_grid(resamples=folds,
            grid=boo_tuneGrid,
            metrics=metric_set(accuracy))



Cuned_nn <- nn_wf %>%
  tune_grid(resamples=folds,
            grid=nn_tuneGrid,
            metrics=metric_set(accuracy)) #Or leave metrics NULL










boost_model <- boost_tree(tree_depth=8,
                          trees=2000,
                          learn_rate=.1) %>%
  set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

boo_wf <- workflow() %>%
  add_recipe(Rec2) %>%
  add_model(boost_model) %>% 
  fit(data=train)

predictions1 <- predict(boo_wf,
                        new_data=test,
                        type="class") %>%  # "class" or "prob" (see doc) 
  bind_cols(., sampleSubmission) %>%
  select(id, .pred_class) %>%
  rename(type=.pred_class)


vroom_write(x=predictions1, file="./boo.csv", delim=",")

bart_model <- bart(trees=tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("classification")

























# Bayes ========================================================================

nb_recipe <- recipe(type ~ ., data = train) %>%
  update_role(id, new_role="id") %>%
  step_dummy(color) %>%
  step_range(all_numeric_predictors(), min=0, max=1) 


bake(prep(nb_recipe), new_data = train)

## nb model
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes eng

nb_model <- naive_Bayes(Laplace=0, smoothness=2) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>%
  add_recipe(nb_recipe) %>%
  add_model(nb_model)%>%
  fit(data=train) 

grid <- grid_regular(Laplace(),
                     smoothness(),
                     levels = 5) ## L^2 total tuning possibilities

## Tune smoothness and Laplace here
folds <- vfold_cv(train, v = 5, repeats=1)

CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=grid,
            metrics=metric_set(accuracy)) #Or leave metrics NULL

## Find best tuning parameters

bestTune <- CV_results %>%
  select_best("accuracy")


## Predict

predictions_nb <- predict(nb_wf,
                          new_data=test,
                          type="class") %>%  # "class" or "prob" (see doc) 
  bind_cols(., sampleSubmission) %>%
  select(id, .pred_class) %>%
  rename(type=.pred_class)


vroom_write(x=predictions_nb, file="./Prediction_f.csv", delim=",")

# SVC===========================================================================

svmPoly <- svm_poly(degree=1, cost=32) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

nb_recipe <- recipe(type ~ ., data = train) %>%
  update_role(id, new_role="id") %>%
  step_dummy(color) %>%
  step_range(all_numeric_predictors(), min=0, max=1) 

svmLinear <- svm_linear(cost=20) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")


svmRadial <- svm_rbf(rbf_sigma=2, cost=4) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(nb_recipe) %>%
  add_model(svmPoly)%>%
  fit(data=train) 

grid <- grid_regular(degree(),
                     cost(),
                     levels = 4) ## L^2 total tuning possibilities

## Tune smoothness and Laplace here
folds <- vfold_cv(train, v = 4, repeats=1)

CV_results <- svm_wf %>%
  tune_grid(resamples=folds,
            grid=grid,
            metrics=metric_set(accuracy)) #Or leave metrics NULL

## Find best tuning parameters

bestTune <- CV_results %>%
  select_best("accuracy")

predictions_s <- predict(svm_wf,
                         new_data=test,
                         type="class") %>%  # "class" or "prob" (see doc) 
  bind_cols(., sampleSubmission) %>%
  select(id, .pred_class) %>%
  rename(type=.pred_class)


vroom_write(x=predictions_s, file="./Prediction_f.csv", delim=",")




# ==============================================================================

Smote_Prep <- rand_forest(mtry= tune(),
                          min_n=tune(),
                          trees=1000) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("classification")

grid <- grid_regular(mtry(range = c(1,9)),
                     min_n(),
                     levels = 6) ## L^2 total tuning possibilities

wf_prep <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(Smote_Prep) 

## Finalize workflow and predict

folds <- vfold_cv(train, v = 5, repeats=1)

CV_results <- wf_prep %>%
  tune_grid(resamples=folds,
            grid=grid,
            metrics=metric_set(accuracy)) #Or leave metrics NULL

## Find best tuning parameters

bestTune <- CV_results %>%
  select_best("accuracy")

nn_recipe <- recipe(type ~ ., data = train) %>%
  update_role(id, new_role="id") %>%
  step_dummy(color) %>%
  step_range(all_numeric_predictors(), min=0, max=1) 


Smote_Model <- rand_forest(mtry = 1,
                           min_n=11,
                           trees=1000) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("classification")

SmoteWF <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(Smote_Model) %>%
  fit(data=train) 

predictions_s <- predict(SmoteWF,
                          new_data=test,
                          type="class") %>%  # "class" or "prob" (see doc) 
  bind_cols(., sampleSubmission) %>%
  select(id, .pred_class) %>%
  rename(type=.pred_class)


vroom_write(x=predictions_s, file="./Prediction_f.csv", delim=",")


