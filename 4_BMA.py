import argparse
import csv
import numpy as np

def compute_bma_weights(evaluation_metrics):
    """
    Compute weights using Bayesian Model Averaging (BMA).

    Parameters:
    - evaluation_metrics (list or ndarray): Evaluation metric values of each base learner.

    Returns:
    - weights (ndarray): Normalized weights based on BMA.
    """
    # Compute posterior probabilities based on evaluation metric values
    posterior_probabilities = np.exp(evaluation_metrics) / np.sum(np.exp(evaluation_metrics))

    # Normalize weights to ensure sum equals one
    weights = posterior_probabilities / np.sum(posterior_probabilities)

    return weights

def read_metrics_from_csv(file_path):
    """
    Read evaluation metrics from the given CSV file.

    Parameters:
    - file_path (str): Path to the input CSV file.

    Returns:
    - metrics (list): List of evaluation metric values read from the file.
    """
    metrics = []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            try:
                for value in row:
                    metrics.append(float(value))
            except ValueError:
                print(f"Skipping invalid value: {value}")
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Compute BMA weights based on evaluation metrics for ensemble learning.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file containing evaluation metrics')
    args = parser.parse_args()

    evaluation_metrics = read_metrics_from_csv(args.input_file)
    
    if not evaluation_metrics:
        print("No valid evaluation metrics found in the input file.")
        return

    #print("Evaluation metrics:", evaluation_metrics)  # Debugging output

    bma_weights = compute_bma_weights(evaluation_metrics)

    # Convert each weight to a standard Python float and round to two decimal places
    formatted_weights = [round(float(w), 2) for w in bma_weights]
    print("BMA weights:", formatted_weights)
    print("Sum of weights:", round(np.sum(bma_weights), 2))  # Should exactly equal 1.00

if __name__ == "__main__":
    main()
