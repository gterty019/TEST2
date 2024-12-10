import numpy as np
from scipy.optimize import differential_evolution
import argparse
import csv

def ensemble_objective(weights, *args):
    """
    Objective function for ensemble learning weights optimization.

    Parameters:
    - weights (list or ndarray): Ensemble weights to be optimized.
    - args: Additional arguments (base_learner_evaluations,).

    Returns:
    - score (float): Negative of the objective function score (e.g., accuracy) to minimize.
    """
    base_learner_evaluations = args[0]
    weighted_score = np.dot(weights, base_learner_evaluations)
    return -weighted_score  # Minimize negative score (maximize score)

def normalize_weights(weights):
    """
    Normalize weights to ensure their sum equals one.

    Parameters:
    - weights (ndarray): Unnormalized weights.

    Returns:
    - normalized_weights (ndarray): Normalized weights.
    """
    total_weight = np.sum(weights)
    normalized_weights = weights / total_weight
    return normalized_weights

def optimize_weights(base_learner_evaluations):
    """
    Optimize ensemble weights using Differential Evolution.

    Parameters:
    - base_learner_evaluations (list or ndarray): Evaluation metric values of each base learner.

    Returns:
    - optimized_weights (ndarray): Optimized ensemble weights with sum equal to one.
    """
    num_base_learners = len(base_learner_evaluations)
    bounds = [(0, 1)] * num_base_learners  # Bounds for weights [0, 1]

    # Differential Evolution optimization
    result = differential_evolution(ensemble_objective, bounds, args=(base_learner_evaluations,))
    optimized_weights = result.x

    # Normalize weights to ensure sum equals one
    normalized_weights = normalize_weights(optimized_weights)

    return normalized_weights

def read_averages_from_csv(file_path):
    """
    Read evaluation metrics from the given CSV file and compute the average for each line.

    Parameters:
    - file_path (str): Path to the input CSV file.

    Returns:
    - averages (list): List of average evaluation metric values computed from the file.
    """
    averages = []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            try:
                values = [float(value) for value in row]
                if values:
                    average_value = sum(values) / len(values)
                    averages.append(average_value)
            except ValueError:
                print(f"Skipping invalid row: {row}")
    return averages

def main():
    parser = argparse.ArgumentParser(description='Optimize ensemble weights using Differential Evolution.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file containing evaluation metrics')
    args = parser.parse_args()

    base_learner_averages = read_averages_from_csv(args.input_file)
    
    if not base_learner_averages:
        print("No valid evaluation metrics found in the input file.")
        return

    print("Base learner averages:", base_learner_averages)  # Debugging output

    optimized_weights = optimize_weights(base_learner_averages)

    # Convert each weight to a standard Python float and round to two decimal places
    formatted_weights = [round(float(w), 2) for w in optimized_weights]
    print("Optimized weights (using DE):", formatted_weights)
    print("Sum of weights:", round(np.sum(optimized_weights), 2))  # Should exactly equal 1.00

if __name__ == "__main__":
    main()