# Load necessary libraries
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

# ----------------------------------------------------------------
# Data Loading and Cleaning
# ----------------------------------------------------------------

load_and_clean_data <- function(filepath) {
  #' Load and clean the drug dataset.
  #'
  #' @param filepath Path to the CSV file.
  #' @return A tibble with cleaned data.

  drugs <- read_csv(filepath) %>%
    select(`Drug Groups`, SMILES, Formula) %>%
    mutate(label = case_when(
      grepl("withdrawn", `Drug Groups`) ~ 0,
      grepl("^approved$", `Drug Groups`) ~ 1,
      TRUE ~ NA_real_
    )) %>%
    drop_na(label) %>%
    filter(!is.na(Formula) & !is.na(SMILES))

  return(drugs)
}

# ----------------------------------------------------------------
# Feature Extraction
# ----------------------------------------------------------------

extract_features <- function(formula, smiles) {
  #' Extract chemical features from formula and SMILES string.
  #'
  #' @param formula The chemical formula.
  #' @param smiles The SMILES string.
  #' @return A tibble with extracted features.

  atomic_weights <- list(
    C = 12.011, H = 1.008, N = 14.007, O = 15.999,
    P = 30.974, S = 32.06, F = 18.998, Cl = 35.45,
    Br = 79.904, I = 126.904
  )
  
  extract_element <- function(formula, element) {
    match <- str_extract(formula, paste0(element, "\\d*"))
    if (!is.na(match)) {
      count <- str_remove(match, element)
      return(ifelse(count == "", 1, as.numeric(count)))
    } else {
      return(0)
    }
  }
  
  C_count <- extract_element(formula, "C")
  H_count <- extract_element(formula, "H")
  N_count <- extract_element(formula, "N")
  O_count <- extract_element(formula, "O")
  P_count <- extract_element(formula, "P")
  S_count <- extract_element(formula, "S")
  F_count <- extract_element(formula, "F")
  Cl_count <- extract_element(formula, "Cl")
  Br_count <- extract_element(formula, "Br")
  I_count <- extract_element(formula, "I")
  
  halogen_count <- F_count + Cl_count + Br_count + I_count
  MW <- (C_count * atomic_weights$C) + (H_count * atomic_weights$H) +
    (N_count * atomic_weights$N) + (O_count * atomic_weights$O) +
    (P_count * atomic_weights$P) + (S_count * atomic_weights$S) +
    (F_count * atomic_weights$F) + (Cl_count * atomic_weights$Cl) +
    (Br_count * atomic_weights$Br) + (I_count * atomic_weights$I)
  HBA <- N_count + O_count
  HBD <- H_count / 2
  total_atoms <- sum(C_count, H_count, N_count, O_count, P_count, S_count, halogen_count)
  
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

add_features_to_data <- function(data) {
  #' Add extracted features to the dataset.
  #'
  #' @param data The input tibble.
  #' @return A tibble with added features.

  data %>%
    mutate(features = map2(Formula, SMILES, extract_features)) %>%
    unnest(features)
}

# ----------------------------------------------------------------
# Data Preprocessing
# ----------------------------------------------------------------

split_data <- function(data) {
  #' Split data into training and testing sets.
  #'
  #' @param data The input tibble.
  #' @return A list containing the training and testing sets.

  set.seed(0)
  split <- initial_split(data, prop = 0.8, strata = label)
  list(
    train = training(split),
    test = testing(split)
  )
}

normalize_features <- function(train_data, test_data) {
  #' Normalize features in the training and testing sets.
  #'
  #' @param train_data The training data.
  #' @param test_data The testing data.
  #' @return A list containing the normalized training and testing sets.

  count_columns <- c("C_count", "H_count", "N_count", "O_count", "P_count", "S_count",
                     "halogen_count", "total_atoms", "ring_count", "single_bonds",
                     "double_bonds", "triple_bonds", "total_bonds")
  continuous_columns <- c("MW", "HBA", "HBD", "complexity_score")

  # Log transform
  train_data <- train_data %>%
    mutate(across(all_of(count_columns), ~ log1p(.x)))
  test_data <- test_data %>%
    mutate(across(all_of(count_columns), ~ log1p(.x)))

  # Scaling parameters from training data
  scaling_means <- colMeans(train_data[c(count_columns, continuous_columns)], na.rm = TRUE)
  scaling_sds <- sapply(train_data[c(count_columns, continuous_columns)], sd, na.rm = TRUE)

  # Standardization
  train_data <- train_data %>%
    mutate(across(all_of(c(count_columns, continuous_columns)),
                  ~ (.x - scaling_means[cur_column()]) / scaling_sds[cur_column()]))
  test_data <- test_data %>%
    mutate(across(all_of(c(count_columns, continuous_columns)),
                  ~ (.x - scaling_means[cur_column()]) / scaling_sds[cur_column()]))

  list(train = train_data, test = test_data)
}

# ----------------------------------------------------------------
# Model Training and Evaluation
# ----------------------------------------------------------------

build_and_train_model <- function(train_features, train_labels, test_features, test_labels) {
  #' Build, train, and evaluate the neural network.
  #'
  #' @param train_features The training features.
  #' @param train_labels The training labels.
  #' @param test_features The testing features.
  #' @param test_labels The testing labels.
  #' @return A list containing the trained model, history, and evaluation.

  input_dim <- ncol(train_features)

  model <- keras_model_sequential() %>%
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

  callbacks <- list(
    callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5, patience = 5, verbose = 1),
    callback_early_stopping(monitor = "val_loss", patience = 10, restore_best_weights = TRUE)
  )

  class_weights <- list(
    "0" = nrow(train_features) / (2 * sum(train_labels == 0)),
    "1" = nrow(train_features) / (2 * sum(train_labels == 1))
  )

  history <- model %>% fit(
    x = as.matrix(train_features),
    y = train_labels,
    epochs = 30,
    batch_size = 32,
    validation_split = 0.3,
    class_weight = class_weights,
    callbacks = callbacks
  )

  evaluation <- model %>% evaluate(
    x = as.matrix(test_features),
    y = test_labels
  )

  list(model = model, history = history, evaluation = evaluation)
}

# ----------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------

main <- function() {
  # Step 1: Load and clean data
  drugs <- load_and_clean_data("approved_drug_structure.csv")

  # Step 2: Feature extraction
  drug_features <- add_features_to_data(drugs)

  # Step 3: Data splitting
  selected_data <- drug_features %>%
    select(-Formula, -SMILES, -`Drug Groups`)

  datasets <- split_data(selected_data)
  train_data <- datasets$train
  test_data <- datasets$test

  # Step 4: Normalize features
  normalized_datasets <- normalize_features(train_data, test_data)
  train_data <- normalized_datasets$train
  test_data <- normalized_datasets$test

  # Step 5: Train and evaluate model
  train_features <- train_data %>% select(-label) %>% select(where(is.numeric))
  train_labels <- train_data$label

  test_features <- test_data %>% select(-label) %>% select(where(is.numeric))
  test_labels <- test_data$label

  results <- build_and_train_model(train_features, train_labels, test_features, test_labels)

  # Print evaluation and plot history
  print("Test Evaluation:")
  print(results$evaluation)
  plot(results$history, metrics = "accuracy")
  plot(results$history, metrics = "auc")
}

# Run the main function
if (sys.nframe() == 0) {
  main()
}
