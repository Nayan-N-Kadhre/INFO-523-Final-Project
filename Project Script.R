
# Import Packages
library(tidyverse)
library(caret)
library(pROC)
library(xgboost)
library(knitr)
library(randomForest)
library(rpart)


# ── EDA AND CLEANING ───────────────────────────────────────────────────────
living_cost <- read.csv("cost_of_living_us.csv", stringsAsFactors = FALSE)
house_data  <- read.csv("realtor-data.zip.csv",  stringsAsFactors = FALSE)

# Lowercase all column names
living_cost <- living_cost %>% rename_all(tolower)
house_data  <- house_data  %>% rename_all(tolower)

# Living cost: split areaname → city / state 
living_cost <- living_cost %>%
  separate(areaname, into = c("city", "state1"), sep = ",") %>%
  mutate(city = trimws(city)) %>%
  select(-any_of(c("case_id", "ismetro", "state1")))

# House data: state abbreviations, drop territories
house_data <- house_data %>%
  mutate(city = trimws(city))

house_data$state[house_data$state == "District of Columbia"] <- "DC"
house_data$state[house_data$state != "DC"] <- state.abb[match(house_data$state[house_data$state != "DC"], state.name)]

house_data <- house_data %>%
  filter(!is.na(state)) %>%
  select(-c("brokered_by", "prev_sold_date"))
# STATUS???
unique(house_data$status)

#Check for Null Values
sum(is.na(living_cost$city))
sum(is.na(living_cost$state))
sum(is.na(house_data$city))
sum(is.na(house_data$state))

# Check presence of duplicate rows
house_data[(duplicated(house_data) | duplicated(house_data, fromLast = TRUE)) & house_data$city == "New Marlborough", ]

# Uniform lowercase city / state 
house_data  <- house_data  %>% mutate(across(c(city, state), tolower))
living_cost <- living_cost %>% mutate(across(c(city, state), tolower))

# Remove duplicate rows 
house_data  <- house_data  %>% distinct()
living_cost <- living_cost %>% distinct()

# Check number of distinct (city, state) pairs in both datasets
house_data %>%
  distinct(city, state) %>%
  nrow()

living_cost %>%
  distinct(city, state) %>%
  nrow()

# Unique city, state pair in the 2 datasets
house_data_pairs <- house_data %>%
  distinct(city, state)

living_cost_pairs <- living_cost %>%
  distinct(city, state)

# Checking if (state, city) pair in the living_cost dataset are present in the housing dataset
matches_housing_income_semi <- living_cost_pairs %>%
  semi_join(house_data_pairs, by = c("city", "state"))

matches_housing_income_anti <- living_cost_pairs %>%
  anti_join(house_data_pairs, by = c("city", "state"))

n_matches <- nrow(matches_housing_income_semi)

n_not_matches <- nrow(matches_housing_income_anti)

n_matches

n_not_matches

# Check Unique family member counts
unique(living_cost$family_member_count)

# # Global plot setttings for figure size and font size
# options(repr.plot.width = 11, repr.plot.height = 11)
# theme_set(theme_minimal(base_size = 20))

# Histogram for distribution of median family income across US states
ggplot(living_cost, aes(x = median_family_income, fill = state)) +
  geom_histogram(aes(colors = state), bins = 20, alpha = 0.7) +
  geom_density() +
  labs(
    title = "Distribution of Median Family Income Across US States",
    x = "Median Family Income",
    y = "Density"
  ) +
  theme_minimal(base_size = 14)

# Check number of records per state
state_counts_living_cost <- living_cost %>%
  count(state, sort = TRUE)

ggplot(state_counts_living_cost, aes(x = reorder(state, n), y = n)) +
  geom_col() +
  coord_flip() +
  labs(title = "Number of Records per State", x = "State", y = "Count")

# Check basic summary statistics for living_cost and house_data
summary(living_cost)
summary(house_data)

# Check structure of both dataset
str(living_cost)
str(house_data)


# pivoting the data to plot histogram and boxplot to check distribution
living_cost_pivot <- living_cost %>%
  pivot_longer(
    cols = where(is.numeric),
    names_to = "variable",
    values_to = "value"
  )

# Histogram of all numeric variables
ggplot(living_cost_pivot, aes(x = value)) +
  geom_histogram(bins = 30) +
  facet_wrap(~ variable, scales = "free") +
  theme_minimal()

# Boxplot of all numeric variables
ggplot(living_cost_pivot, aes(y = value)) +
  geom_boxplot() +
  facet_wrap(~ variable, scales = "free") +
  theme_minimal()

# Check presence of null values in both dataset
colSums(is.na(living_cost))
colSums(is.na(house_data))

# Impute missing median_family_income with column mean
living_cost <- living_cost %>%
  mutate(median_family_income = ifelse(
    is.na(median_family_income),
    mean(median_family_income, na.rm = TRUE),
    median_family_income
  ))

# 20% of the data has chilcare cost as 0 (6286 rows)
sum(living_cost$childcare_cost == 0, na.rm = TRUE) / dim(living_cost[0]) * 100

# That 20% are those families with No children (family member count = 1p0c or 2p0c)
living_cost %>%
  count(family_member_count) %>%
  mutate(
    percentage = n / sum(n) * 100
  )

view(house_data)

dim(house_data)
dim(living_cost)




# ── FEATURE ENGINEERING  ──────────────────────────────────────────────────

# Derived columns
living_cost <- living_cost %>%
  mutate(
    available_income     = median_family_income - total_cost,
    affordable_flag      = if_else(available_income > 0, 1L, 0L), # 1 = affordable city
    income_to_cost_ratio = median_family_income / total_cost,
    housing_pct          = housing_cost / total_cost,
    childcare_pct        = childcare_cost / total_cost,
    tax_pct              = taxes / total_cost
  )

# House data: drop rows missing critical fields 


# Meaningful loss: as these are important fields to predit and and imputing would mean inventing facts about the preperties which may or may not be true and training on this data is not reliable to produces real and accurate results.

nrow(house_data)

colSums(is.na(house_data))

str(house_data)

house_data <- house_data %>%
  filter(
    !is.na(price),
    !is.na(bed),
    !is.na(bath),
    !is.na(acre_lot),
    !is.na(street),
    !is.na(house_size),
    !is.na(zip_code),
    price      > 0,
    house_size > 0,
    bed        > 0
  ) %>%
  mutate(
    price_per_sqft = price / house_size,
    log_price      = log1p(price),
    log_size       = log1p(house_size)
  )

cat("Living cost rows:", nrow(living_cost), " | House data rows:", nrow(house_data), "\n")

view(house_data)

# State Level aggregation
# Since the two datasets cannot be merged on a primary key, all cross-dataset, insights are derived at the STATE level.

# Living cost - state level summaries
state_income <- living_cost %>%
  group_by(state) %>%
  summarise(
    median_income          = median(median_family_income, na.rm = TRUE),
    median_available_inc   = median(available_income,     na.rm = TRUE),
    median_total_cost      = median(total_cost,           na.rm = TRUE),
    median_housing_cost    = median(housing_cost,         na.rm = TRUE),
    pct_affordable_cities  = mean(affordable_flag,        na.rm = TRUE) * 100,
    avg_income_cost_ratio  = mean(income_to_cost_ratio,   na.rm = TRUE),
    n_cities_lc            = n_distinct(city),
    .groups = "drop"
  )

# House data - state level summaries
state_housing <- house_data %>%
  group_by(state) %>%
  summarise(
    median_price        = median(price,          na.rm = TRUE),
    median_price_sqft   = median(price_per_sqft, na.rm = TRUE),
    median_house_size   = median(house_size,     na.rm = TRUE),
    median_beds         = median(bed,            na.rm = TRUE),
    n_listings          = n(),
    .groups = "drop"
  )


view(state_income)
view(state_housing)


# Merge - join at state level
state_bridge <- state_income %>%
  inner_join(state_housing, by = "state") %>%
  mutate(
    # How many years of available income needed to buy median house
    years_to_afford = if_else(median_available_inc > 0, median_price / median_available_inc, NA_real_),
    # Standard 28% front-end DTI: max annual mortgage ≈ 0.28 × gross income
    # Rough house budget using 30-yr @ ~7% → monthly ≈ price × 0.00665
    # annual mortgage ≈ price × 0.0798 ; affordable if ≤ 0.28 × income
    max_affordable_price = (0.28 * median_income) / 0.0798,
    affordability_gap    = median_price - max_affordable_price,
    is_affordable_state  = if_else(affordability_gap <= 0, "Affordable", "Unaffordable")
  )

view(state_bridge)
dim(state_bridge)
colnames(state_bridge)

# living_cost
# Split family_member_count into separate numeric cols for number of adults and childern
living_cost <- living_cost %>%
  mutate(
    adults   = as.integer(substr(family_member_count, 1, 1)),
    children = as.integer(substr(family_member_count, 3, 3))
  )

# house_data
# Binning house size (sqft) and price for classification
house_data <- house_data %>%
  mutate(
    size_tier = cut(
      house_size,
      breaks = c(0, 1000, 1500, 2000, 3000, Inf),
      labels = c("Tiny(<1k)", "Small(1-1.5k)", "Medium(1.5-2k)", "Large(2-3k)", "XLarge(3k+)"),
      right  = TRUE
    ),
    price_tier = cut(
      price,
      breaks = c(0, 150000, 300000, 500000, 750000, Inf),
      labels = c("Budget(<150k)", "Affordable(150-300k)", "Mid(300-500k)",
                 "Premium(500-750k)", "Luxury(750k+)"),
      right  = TRUE
    ),
    bath_int = floor(bath)   # whole bathrooms only for modelling
  )

# computes each state's median price-per-sqft, then join that state benchmark onto every individual house listing in house_data
state_ppsf <- house_data %>%
  group_by(state) %>%
  summarise(state_median_ppsf = median(price_per_sqft, na.rm = TRUE), .groups = "drop")

house_data <- house_data %>% left_join(state_ppsf, by = "state")

view(house_data)





# ── SCALING ────────────────────────────────────────────────────────────────

robust_scale <- function(x) {
  med <- median(x, na.rm = TRUE)
  iqr <- IQR(x, na.rm = TRUE)
  if (iqr == 0) return(x - med)   
  (x - med) / iqr
}

# Living cost: scale numeric predictors (keep originals for interpretation)
lc_numeric_cols <- c(
  "housing_cost", "food_cost", "transportation_cost",
  "healthcare_cost", "other_necessities_cost", "childcare_cost",
  "taxes", "total_cost", "median_family_income",
  "adults", "children", "income_to_cost_ratio",
  "housing_pct", "childcare_pct", "tax_pct"
)

living_cost_scaled <- living_cost %>%
  mutate(across(all_of(lc_numeric_cols), robust_scale, .names = "{.col}_s"))

# House data: scale numeric predictors
hd_numeric_cols <- c("bed", "bath_int", "house_size", "acre_lot",
                     "price_per_sqft", "state_median_ppsf")

house_data_scaled <- house_data %>%
  mutate(across(all_of(hd_numeric_cols), robust_scale, .names = "{.col}_s"))




# CLASSIFICATION - Whether a specific family configuration (adults + children + their cost breakdown) can afford to live in a given city?

# City/state excluded — scaled cost columns implicitly encode geography;
# the model learns cost patterns, not memorized location names.
lc_features <- c(
  "housing_cost_s", "food_cost_s", "transportation_cost_s",
  "healthcare_cost_s", "other_necessities_cost_s", "childcare_cost_s",
  "taxes_s", "adults", "children"
)

lc_model_data <- living_cost_scaled %>%
  select(all_of(lc_features), affordable_flag) %>%
  mutate(affordable_flag = factor(affordable_flag,
                                  levels = c(0, 1),
                                  labels = c("No", "Yes"))) %>%
  drop_na()

# Stratified Train-test split 
train_idx <- createDataPartition(lc_model_data$affordable_flag,
                                 p = 0.75, list = FALSE)
lc_train <- lc_model_data[train_idx, ]
lc_test  <- lc_model_data[-train_idx, ]

# Configuration for Training all the models (5-fold CV)
ctrl_cls <- trainControl(
  method = "cv", 
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# Logistic Regression 
set.seed(42)
fit_lr <- train(
  affordable_flag ~ .,
  data      = lc_train,
  method    = "glm",
  family    = "binomial",
  metric    = "ROC",
  trControl = ctrl_cls
)

# Decision Tree
set.seed(42)
fit_dt <- train(
  affordable_flag ~ .,
  data      = lc_train,
  method    = "rpart",
  metric    = "ROC",
  trControl = ctrl_cls,
  tuneLength = 10
)

# Random Forest
set.seed(42)
fit_rf_cls <- train(
  affordable_flag ~ .,
  data      = lc_train,
  method    = "rf",
  metric    = "ROC",
  trControl = ctrl_cls,
  tuneLength = 5,
  ntree      = 300
)

# Evaluate on held-out test set for 
eval_classifier <- function(model, test_data, model_name) {
  preds <- predict(model, newdata = test_data)
  probs <- predict(model, newdata = test_data, type = "prob")[, "Yes"]
  truth <- test_data$affordable_flag
  cm <- confusionMatrix(preds, truth, positive = "Yes")
  roc_obj <- pROC::roc(truth, probs, levels = c("No", "Yes"), quiet = TRUE)
  
  tibble(
    Model     = model_name,
    Accuracy  = round(cm$overall["Accuracy"], 4),
    Precision = round(cm$byClass["Precision"], 4),
    Recall    = round(cm$byClass["Recall"], 4),
    F1        = round(cm$byClass["F1"], 4),
    AUC_ROC   = round(pROC::auc(roc_obj), 4)
  )
}

# Run the evaluate function on the three models
library(pROC)
cls_results <- bind_rows(
  eval_classifier(fit_lr,     lc_test, "Logistic Regression"),
  eval_classifier(fit_dt,     lc_test, "Decision Tree"),
  eval_classifier(fit_rf_cls, lc_test, "Random Forest")
)

# Classification Results for all the models
print(cls_results, n = Inf)





# REGRESSION — How much annual income a house listing requires for a family to afford it?
house_data_scaled <- house_data_scaled %>%
  mutate(
    log_required_income = log(price / 0.28)
  )

house_features <- c(
  "bed", "bath_int_s", "house_size_s", "acre_lot_s",
  "log_size", "state_median_ppsf_s"
)

# NOTE: log_price IS used here because at prediction time we know the listing
# price — predicting required income FROM the listing, not from latent features.

house_model_data <- house_data_scaled %>%
  select(all_of(house_features), log_required_income) %>%
  drop_na() %>%
  filter(if_all(everything(), is.finite))  # drop Inf/NaN in ALL columns

# Representative 50% sample → stratified 80/20 split
set.seed(42)
hd_sample <- house_model_data %>% slice_sample(prop = 0.50)

train_idx_hd <- createDataPartition(hd_sample$log_required_income,
                                    p = 0.80, list = FALSE)
hd_train <- hd_sample[train_idx_hd, ]
hd_test  <- hd_sample[-train_idx_hd, ]

# Verify no bad values made it through
stopifnot(all(is.finite(hd_train$log_required_income)))
stopifnot(all(is.finite(hd_test$log_required_income)))

# Configuration for training regression models
ctrl_reg <- trainControl(
  method          = "cv",
  number          = 5,
  savePredictions = "final"
)

# Linear Regression
set.seed(42)
fit_lm <- train(
  log_required_income ~ .,
  data      = hd_train,
  method    = "lm",
  trControl = ctrl_reg
)

# Prepare matrices for xgboost
X_train <- as.matrix(hd_train[, house_features])
y_train <- hd_train$log_required_income
X_test  <- as.matrix(hd_test[, house_features])
y_test  <- hd_test$log_required_income

dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest  <- xgb.DMatrix(data = X_test,  label = y_test)

# Gradient Boosting via xgboost directly
set.seed(42)
fit_gbm <- xgb.train(
  params = list(
    objective        = "reg:squarederror",
    eta              = 0.3,
    max_depth        = 3,
    subsample        = 0.75,
    colsample_bytree = 0.8
  ),
  data    = dtrain,
  nrounds = 150,
  verbose = 0
)

# Define evaluate function for Regression models
eval_regressor <- function(model, test_data, model_name, is_xgb = FALSE) {
  if (is_xgb) {
    preds <- predict(model, newdata = xgb.DMatrix(as.matrix(test_data[, house_features])))
  } else {
    preds <- predict(model, newdata = test_data)
  }
  truth <- test_data$log_required_income
  n <- length(truth)
  k <- length(house_features)
  
  rmse_val <- sqrt(mean((truth - preds)^2))
  mae_val <- mean(abs(truth - preds))
  ss_res <- sum((truth - preds)^2)
  ss_tot <- sum((truth - mean(truth))^2)
  r2_val <- 1 - ss_res / ss_tot
  adj_r2 <- 1 - (1 - r2_val) * (n - 1) / (n - k - 1)
  
  tibble(
    Model = model_name,
    RMSE = round(rmse_val, 5),
    MAE = round(mae_val, 5),
    R2 = round(r2_val, 5),
    Adj_R2 = round(adj_r2, 5)
  )
}

reg_results <- bind_rows(
  eval_regressor(fit_lm,  hd_test, "Linear Regression"),
  eval_regressor(fit_gbm, hd_test, "Gradient Boosting (XGB)", is_xgb = TRUE)
)

print(reg_results)







# Classification Results
print(cls_results)

# AUC-ROC curves 
cls_models  <- list(fit_lr, fit_dt, fit_rf_cls)
cls_names   <- c("Logistic Regression", "Decision Tree", "Random Forest")
cls_colors  <- c("#e41a1c", "#377eb8", "#4daf4a")

for (i in seq_along(cls_models)) {
  probs   <- predict(cls_models[[i]], newdata = lc_test, type = "prob")[, "Yes"]
  roc_obj <- pROC::roc(lc_test$affordable_flag, probs,
                       levels = c("No", "Yes"), quiet = TRUE)
  auc_val <- round(as.numeric(pROC::auc(roc_obj)), 3)
  if (i == 1) {
    plot(roc_obj, col = cls_colors[i], lwd = 2,
         main = "ROC Curves — City Affordability Classification")
  } else {
    plot(roc_obj, col = cls_colors[i], lwd = 2, add = TRUE)
  }
  cls_names[i] <- paste0(cls_names[i], " (AUC = ", auc_val, ")")
}
legend("bottomright", legend = cls_names, col = cls_colors, lwd = 2, cex = 0.9)

# Decision tree plot 
rpart.plot(
  fit_dt$finalModel,
  type          = 4,
  extra         = 104,
  fallen.leaves = TRUE,
  main          = "Decision Tree — Is this city affordable?"
)

# Variable importance — best classification model (highest AUC)
best_cls_model <- cls_models[[which.max(cls_results$AUC_ROC)]]
best_cls_name  <- cls_results$Model[which.max(cls_results$AUC_ROC)]

varImp(best_cls_model)$importance %>%
  rownames_to_column("Feature") %>%
  arrange(desc(Overall)) %>%
  ggplot(aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_col(fill = "#2c7bb6", width = 0.7) +
  coord_flip() +
  labs(
    title    = "Variable Importance — City Affordability",
    subtitle = paste("Model:", best_cls_name),
    x        = NULL,
    y        = "Importance"
  ) +
  theme_minimal(base_size = 13)

#  State summary
lc_features <- c(
  "housing_cost_s", "food_cost_s", "transportation_cost_s",
  "healthcare_cost_s", "other_necessities_cost_s", "childcare_cost_s",
  "taxes_s", "adults", "children"
)

city_affordability <- living_cost_scaled %>%
  select(state, city, all_of(lc_features), affordable_flag) %>%
  drop_na() %>%
  mutate(
    predicted_label = predict(best_cls_model,
                              newdata = living_cost_scaled %>%
                                select(all_of(lc_features)) %>%
                                drop_na()),
    predicted_prob  = predict(best_cls_model,
                              newdata = living_cost_scaled %>%
                                select(all_of(lc_features)) %>%
                                drop_na(),
                              type = "prob")[, "Yes"]
  )

t1_state_summary <- city_affordability %>%
  group_by(state) %>%
  summarise(
    total_cities         = n(),
    pct_predicted_afford = round(mean(predicted_label == "Yes") * 100, 1),
    avg_afford_prob      = round(mean(predicted_prob), 3),
    .groups = "drop"
  )

 # Affordable cities by state
t1_state_summary %>%
  ggplot(aes(x = reorder(state, pct_predicted_afford),
             y = pct_predicted_afford,
             fill = pct_predicted_afford)) +
  geom_col(width = 0.7) +
  coord_flip() +
  scale_fill_gradient(low = "#d73027", high = "#1a9850", name = "% Affordable") +
  labs(
    title    = "% of Cities Predicted Affordable by State",
    subtitle = "Based on living cost vs income classification",
    x        = NULL,
    y        = "% Cities Affordable"
  ) +
  theme_minimal(base_size = 11)









# Regression Results
print(reg_results)

# Actual vs predicted - Linear Regression
lm_preds <- predict(fit_lm, newdata = hd_test)

hd_test %>%
  sample_n(min(50000, nrow(hd_test))) %>%
  mutate(predicted = predict(fit_lm, newdata = .)) %>%
  ggplot(aes(x = log_required_income, y = predicted)) +
  geom_point(alpha = 0.2, color = "#2c7bb6", size = 0.8) +
  geom_abline(slope = 1, intercept = 0, color = "#d73027", lwd = 1) +
  labs(
    title    = "Actual vs Predicted — Linear Regression",
    subtitle = "Red line = perfect prediction | log scale",
    x        = "Actual log(Required Income)",
    y        = "Predicted log(Required Income)"
  ) +
  theme_minimal(base_size = 13)

# ctual vs predicted - XGBoost 
gbm_preds <- predict(fit_gbm,
                     newdata = xgb.DMatrix(as.matrix(hd_test[, house_features])))

hd_test %>%
  sample_n(min(50000, nrow(hd_test))) %>%
  mutate(
    predicted = predict(fit_gbm,
                        newdata = xgb.DMatrix(as.matrix(.[, house_features])))
  ) %>%
  ggplot(aes(x = log_required_income, y = predicted)) +
  geom_point(alpha = 0.2, color = "#4daf4a", size = 0.8) +
  geom_abline(slope = 1, intercept = 0, color = "#d73027", lwd = 1) +
  labs(
    title    = "Actual vs Predicted — Gradient Boosting (XGBoost)",
    subtitle = "Red line = perfect prediction | log scale",
    x        = "Actual log(Required Income)",
    y        = "Predicted log(Required Income)"
  ) +
  theme_minimal(base_size = 13)

# XGBoost variable importance
xgb.importance(model = fit_gbm) %>%
  as_tibble() %>%
  ggplot(aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "#4daf4a", width = 0.7) +
  coord_flip() +
  labs(
    title    = "Variable Importance — Gradient Boosting (XGBoost)",
    subtitle = "What income does this house require?",
    x        = NULL,
    y        = "Gain"
  ) +
  theme_minimal(base_size = 13)
