#!/bin/bash

# Define the dataset code (this is the only variable you need to change)
dataset_code_number="D1_LEAKDB"
#D1_LEAKDB
#D2_NSL_KDD
#D3_SG_MiTM
#D4_CICSIDS2017
# Automatically derive other file names based on dataset_code_number
input_file="${dataset_code_number}.csv"
mid_file="${dataset_code_number}_ACC.csv"
output_file="${dataset_code_number}_ACCURACIES.csv"

# Print to check
echo "Dataset Code: $dataset_code_number"
echo "Input File: $input_file"
echo "Mid File: $mid_file"
echo "Output File: $output_file"

weights_folder="WEIGHTS"
results_folder="RESULTS"
weight_scripts_folder="WEIGHTS_SCRIPTS"
accuracy_scripts_folder="MATRIX_CSV_SCRIPTS"
inference_scripts_folder="INFERENCE_SCRIPTS"

# Ensure the results folder exists
mkdir -p $weights_folder
mkdir -p $results_folder

# Computing Weights Using Uniform Weighting Scheme
#echo "Stratified k-Folds (k=5) Using a Set of Classifiers"
#python3 "$accuracy_scripts_folder/0_1_STRATIFIED_KFOLDS.py" $input_file $mid_file
#echo "Exclude Unnecessary lines from the middle file (produced) "
#python3 "$accuracy_scripts_folder/0_2_EXCLUDE_UNNEED.py" $mid_file $output_file
#rm  $mid_file

echo "Uniform Weighting Per Algorithm (UW)"
python3 "$weight_scripts_folder/1_UW.py" $output_file > "$weights_folder/UW_${dataset_code_number}.txt"

echo "Uniform Weighting Per Algorithm Class (UW_PCA)"
python3 "$weight_scripts_folder/1_UW_PCA.py" $output_file > "$weights_folder/UWPCA_${dataset_code_number}.txt"

echo "Weighting Given Accuracy Per Algorithm (WA)"
python3 "$weight_scripts_folder/2_WA.py" $output_file > "$weights_folder/WA_${dataset_code_number}.txt"

echo "Weighting Given Accuracy Per Algorithm (WA_PCA)"
python3 "$weight_scripts_folder/2_WA_PCA.py" $output_file > "$weights_folder/WAPCA_${dataset_code_number}.txt"

echo "Differential Evaluation Weighting Per Algorithm (DE)"
python3 "$weight_scripts_folder/3_DE.py" $output_file > "$weights_folder/DE_${dataset_code_number}.txt"

echo "Bayesian Model Averaging Weighting Per Algorithm Class (BMA)"
python3 "$weight_scripts_folder/4_BMA.py" $output_file > "$weights_folder/BMA_${dataset_code_number}.txt"

echo "MIP & Elastic Net Weighting Per Algorithm Class (MIPEN)"
python3 "$weight_scripts_folder/5_MIP.py" $output_file > "$weights_folder/MIPEN_${dataset_code_number}.txt"

# Computing Inference Using Various Weighting Schemes
#python3 "$inference_scripts_folder/2_INFERENCE_ALGORITHM.py" --dataset $input_file --weights 0.12,0.12,0.12,0.12,0.12,0.12,0.12,0.12 --output "$results_folder/1_NSL_UW.csv"
#python3 "$inference_scripts_folder/2_INFERENCE_ALGORITHM.py" --dataset $input_file --weights 0.15,0.13,0.13,0.13,0.12,0.12,0.12,0.12 --output "$results_folder/1_NSL_WA.csv"
#python3 "$inference_scripts_folder/2_INFERENCE_ALGORITHM.py" --dataset $input_file --weights 0.12,0.12,0.12,0.12,0.12,0.12,0.12,0.12 --output "$results_folder/1_NSL_DE.csv"
#python3 "$inference_scripts_folder/2_INFERENCE_ALGORITHM_CLASS.py"  --dataset $input_file --weights 0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02  --output "$results_folder/1_NSL_UW_PCA.csv"
#python3 "$inference_scripts_folder/2_INFERENCE_ALGORITHM_CLASS.py"  --dataset $input_file --weights 0.04,0.02,0.02,0.04,0.04,0.02,0.02,0.04,0.02,0.04,0.04,0.02,0.02,0.04,0.04,0.02,0.02,0.04,0.02,0.04,0.04,0.02,0.02,0.04,0.04,0.02,0.02,0.04,0.02,0.04,0.01,0.01  --output "$results_folder/1_NSL_WA_PCA.csv"
#python3 "$inference_scripts_folder/2_INFERENCE_ALGORITHM_CLASS.py"  --dataset $input_file --weights 0.15,0.24,0.39,0.15,0.15,0.00,0.21,0.15,0.15,0.05,0.08,0.15,0.06,0.51,0.00,0.00,0.15,0.00,0.06,0.19,0.15,0.00,0.03,0.19,0.15,0.00,0.00,0.19,0.06,0.20,0.24,0.00 --output "$results_folder/1_NSL_MIPEN.csv"
#python3 "$inference_scripts_folder/2_INFERENCE_ALGORITHM_CLASS.py"  --dataset $input_file --weights 0.04,0.02,0.02,0.04,0.04,0.02,0.02,0.04,0.04,0.02,0.02,0.04,0.04,0.03,0.02,0.03,0.04,0.02,0.02,0.04,0.04,0.02,0.02,0.04,0.04,0.02,0.02,0.04,0.04,0.02,0.02,0.02 --output "$results_folder/1_NSL_BMA.csv"

# End of script
echo "Script execution completed."