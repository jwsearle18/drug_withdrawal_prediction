library(tidyverse)
library(purrr)
library(progress)
library(tidymodels)
library(keras)
library(yardstick)
library(rcdk)
library(ranger)
library(kernlab)
library(xgboost)
library(themis)


#####################################################
# STEP 1 - CLEAN DATA
#####################################################

## LOAD DATASET
drugs = read_csv("approved_drug_structure.csv")
print(head(drugs))
print(nrow(drugs))

## SELECT RELEVANT COLUMNS
drugs = drugs %>%
  select(`Drug Groups`, SMILES, Formula)
print(head(drugs))

## LABELS FOR 'APPROVED' AND 'WITHDRAWN'
drugs = drugs %>%
  mutate(label = case_when(
    grepl("withdrawn", `Drug Groups`) ~ 0,
    grepl("^approved$", `Drug Groups`) ~ 1,
    TRUE ~ NA_real_
  )) %>%
  drop_na(label)

print(table(drugs$label))
print(nrow(drugs))

## REMOVE MISSING ROWS
drugs = drugs %>%
  filter(!is.na(Formula) & !is.na(SMILES))

print(head(drugs))
print(nrow(drugs))
View(drugs)

#####################################################
# STEP 2 - FEATURE EXTRACTION
#####################################################

extract_features <- function(formula, smiles) {
  atomic_weights = list(
    C = 12.011, H = 1.008, N = 14.007, O = 15.999,
    P = 30.974, S = 32.06, F = 18.998, Cl = 35.45,
    Br = 79.904, I = 126.904
  )
  
  extract_element = function(formula, element) {
    match = str_extract(formula, paste0(element, "\\d*"))
    if (!is.na(match)) {
      count = str_remove(match, element)
      return(ifelse(count == "", 1, as.numeric(count)))
    } else {
      return(0)
    }
  }
  
  C_count = extract_element(formula, "C")
  H_count = extract_element(formula, "H")
  N_count = extract_element(formula, "N")
  O_count = extract_element(formula, "O")
  P_count = extract_element(formula, "P")
  S_count = extract_element(formula, "S")
  F_count = extract_element(formula, "F")
  Cl_count = extract_element(formula, "Cl")
  Br_count = extract_element(formula, "Br")
  I_count = extract_element(formula, "I")
  
  halogen_count = F_count + Cl_count + Br_count + I_count
  MW = (C_count * atomic_weights$C) + (H_count * atomic_weights$H) +
    (N_count * atomic_weights$N) + (O_count * atomic_weights$O) +
    (P_count * atomic_weights$P) + (S_count * atomic_weights$S) +
    (F_count * atomic_weights$F) + (Cl_count * atomic_weights$Cl) +
    (Br_count * atomic_weights$Br) + (I_count * atomic_weights$I)
  HBA = N_count + O_count
  HBD = H_count / 2
  total_atoms = sum(C_count, H_count, N_count, O_count, P_count, S_count, halogen_count)
  
  ring_count <- str_count(smiles, "\\d")
  single_bonds <- str_count(smiles, "-")
  double_bonds <- str_count(smiles, "=")
  triple_bonds <- str_count(smiles, "#")
  total_bonds <- single_bonds + double_bonds + triple_bonds
  has_ring <- ifelse(ring_count > 0, 1, 0)
  complexity_score <- total_bonds + ring_count
  
  return(tibble(
    C_count = C_count, H_count = H_count, N_count = N_count, O_count = O_count,
    P_count = P_count, S_count = S_count, halogen_count = halogen_count, 
    total_atoms = total_atoms, MW = MW, HBA = HBA, HBD = HBD,
    ring_count = ring_count, single_bonds = single_bonds, double_bonds = double_bonds,
    triple_bonds = triple_bonds,
    total_bonds = total_bonds, has_ring = has_ring, complexity_score = complexity_score
  ))
}

drug_features <- drugs %>%
  mutate(features = map2(Formula, SMILES, extract_features)) %>%
  unnest(features)

print(head(drug_features))
View(drug_features)

#####################################################
# STEP 3 - DATA SPLIT
#####################################################
print("Class Distribution Before Splitting:")
print(table(drug_features$label))

set.seed(0)

## DROP UNUSED COLUMNS
selected_data = drug_features %>%
  select(-Formula, -SMILES, -`Drug Groups`)

## SPLIT DATA WITH STRATIFIED SAMPLING
split = initial_split(selected_data, prop = 0.8, strata = label)

## CREATE TRAIN AND TEST DATA
train_data = training(split)
test_data = testing(split)

## CHECK SPLIT
print(nrow(train_data))
print(nrow(test_data))
print(table(train_data$label))
print(table(test_data$label))

print("Class Distribution in Training Data:")
print(table(train_data$label))

print("Class Distribution in Test Data:")
print(table(test_data$label))


View(train_data)

#####################################################
# STEP 4 - NORMALIZE FEATURES
#####################################################

train_data = as_tibble(train_data)
test_data = as_tibble(test_data)

binary_columns = c("has_ring")
count_columns = c("C_count", "H_count", "N_count", "O_count", "P_count", "S_count",
                  "halogen_count", "total_atoms", "ring_count", "single_bonds",
                  "double_bonds", "triple_bonds", "total_bonds")
continuous_columns = c("MW", "HBA", "HBD", "complexity_score")

## LOG TRANSFORM
train_data = train_data %>%
  mutate(across(all_of(count_columns), ~ log1p(.x)))

test_data = test_data %>%
  mutate(across(all_of(count_columns), ~ log1p(.x)))

## SCALING PARAMS
scaling_means = colMeans(train_data[c(count_columns, continuous_columns)], na.rm = TRUE)
scaling_sds = sapply(train_data[c(count_columns, continuous_columns)], sd, na.rm = TRUE)

## STANDARDIZATION
train_data = train_data %>%
  mutate(across(all_of(c(count_columns, continuous_columns)),
                ~ (.x - scaling_means[cur_column()]) / scaling_sds[cur_column()]))

test_data = test_data %>%
  mutate(across(all_of(c(count_columns, continuous_columns)),
                ~ (.x - scaling_means[cur_column()]) / scaling_sds[cur_column()]))

print(head(train_data))
View(train_data)


#####################################################
# STEP 5 - TRAIN NEURAL NETWORK
#####################################################

train_features = train_data %>%
  select(-label) %>%
  select(where(is.numeric))

train_labels = train_data$label

test_features = test_data %>%
  select(-label) %>%
  select(where(is.numeric))

test_labels = test_data$label

input_dim = ncol(train_features)

model = keras_model_sequential() %>%
  layer_dense(units = 32, activation = "leaky_relu", input_shape = input_dim) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 16, activation = "leaky_relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "binary_crossentropy",
  metrics = c("accuracy", 
              keras::metric_auc(name = "auc"), 
              keras::metric_precision(name = "precision"), 
              keras::metric_recall(name = "recall"))
)

callbacks = list(
  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5, patience = 5, verbose = 1),
  callback_early_stopping(monitor = "val_loss", patience = 10, restore_best_weights = TRUE)
)

class_weights <- list(
  "0" = nrow(train_data) / (2 * sum(train_data$label == 0)),
  "1" = nrow(train_data) / (2 * sum(train_data$label == 1))
)

history = model %>% fit(
  x = as.matrix(train_features),
  y = train_labels,
  epochs = 30,
  batch_size = 32,
  validation_split = 0.3,
  class_weight = class_weights,
  callbacks = callbacks
)

evaluation = model %>% evaluate(
  x = as.matrix(test_features),
  y = test_labels
)
print("Test Evaluation:")
print(evaluation)

plot(history, metrics = "accuracy")
plot(history, metrics = "auc")







