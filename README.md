# Drug Withdrawal Prediction

This project uses a neural network to predict whether a drug will be withdrawn from the market based on its chemical properties. The model is built in R using the Keras library.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Introduction

The goal of this project is to predict drug withdrawal from the market using molecular features. Drug withdrawal can occur for various reasons, including safety concerns, lack of efficacy, or commercial reasons. This project focuses on predicting withdrawal based on the chemical structure of the drug.

## Dataset

The dataset used in this project is `approved_drug_structure.csv`, which contains information about approved and withdrawn drugs. The key features used for prediction are the drug's SMILES (Simplified Molecular Input Line Entry System) string and its chemical formula.

- **SMILES:** A string representation of the chemical structure of a molecule.
- **Formula:** The chemical formula of the drug.
- **Drug Groups:** Indicates whether the drug is approved or has been withdrawn.

## Methodology

The project follows these steps:

1.  **Data Cleaning:** The dataset is loaded, and relevant columns are selected. Drugs are labeled as "approved" (1) or "withdrawn" (0). Missing values are removed.
2.  **Feature Extraction:** Chemical features are extracted from the SMILES strings and chemical formulas. These features include:
    *   Counts of different atoms (C, H, N, O, P, S, halogens)
    *   Molecular Weight (MW)
    *   Number of Hydrogen Bond Acceptors (HBA) and Donors (HBD)
    *   Ring count, bond counts (single, double, triple), and a complexity score.
3.  **Data Preprocessing:**
    *   The data is split into training and testing sets with stratified sampling to maintain the class distribution.
    *   The features are normalized using log transformation and standardization.
4.  **Model Training:**
    *   A neural network is built using Keras with a sequential model architecture.
    *   The model is compiled with the Adam optimizer and binary cross-entropy loss function.
    *   Class weights are used to handle the class imbalance between approved and withdrawn drugs.
    *   The model is trained with early stopping and learning rate reduction on plateau to prevent overfitting.
5.  **Evaluation:** The model's performance is evaluated on the test set using metrics like accuracy, AUC, precision, and recall.

## Installation

To run this project, you need to have R and the following packages installed. You can install them using the following command in your R console:

```R
install.packages(c("tidyverse", "purrr", "progress", "tidymodels", "keras", "yardstick", "rcdk", "ranger", "kernlab", "xgboost", "themis"))
```

## Usage

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/drug_withdrawal_prediction.git
    cd drug_withdrawal_prediction
    ```
2.  Run the R script:
    ```bash
    Rscript drug_withdrawal_prediction.R
    ```

## Results

The script will print the model's performance on the test set. The training history, including accuracy and AUC plots, will also be displayed. The `Report.docx` file contains a more detailed report of the project and its findings.
