import argparse
import csv

def compute_accuracy_based_weights(accuracies):
    """
    Compute weights based on individual accuracies for ensemble learning.

    Parameters:
    - accuracies (list): List of accuracies from each base learner.

    Returns:
    - weights (list): List of normalized weights based on accuracies.
    """
    total_accuracy = sum(accuracies)
    weights = [acc / total_accuracy for acc in accuracies]
    return weights

def read_accuracies_from_csv(file_path):
    """
    Read accuracies from the given CSV file.

    Parameters:
    - file_path (str): Path to the input CSV file.

    Returns:
    - accuracies (list): List of accuracies read from the file.
    """
    accuracies = []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            try:
                for value in row:
                    accuracies.append(float(value))
            except ValueError:
                print(f"Skipping invalid value: {value}")
    return accuracies

def main():
    parser = argparse.ArgumentParser(description='Compute weights based on individual accuracies for ensemble learning.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file containing accuracies')
    args = parser.parse_args()
    
    accuracies = read_accuracies_from_csv(args.input_file)
    
    if not accuracies:
        print("No valid accuracies found in the input file.")
        return
    
    accuracy_based_weights = compute_accuracy_based_weights(accuracies)

    # Print weights rounded to two decimal places
    formatted_weights = [round(w, 2) for w in accuracy_based_weights]
    print("Accuracy-based weights:", formatted_weights)
    print("Sum of weights:", round(sum(accuracy_based_weights), 2))  # Should equal 1.00

if __name__ == "__main__":
    main()
