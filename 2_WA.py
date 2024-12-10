import argparse
import csv

def compute_accuracy_based_weights(average_accuracies):
    """
    Compute weights based on average accuracy for ensemble learning.

    Parameters:
    - average_accuracies (list): List of average accuracies of each base learner.

    Returns:
    - weights (list): List of normalized weights based on average accuracy.
    """
    total_accuracy = sum(average_accuracies)
    weights = [acc / total_accuracy for acc in average_accuracies]
    return weights

def read_accuracies_from_csv(file_path):
    """
    Read accuracies from the given CSV file and compute average accuracy for each line.

    Parameters:
    - file_path (str): Path to the input CSV file.

    Returns:
    - average_accuracies (list): List of average accuracies computed from the file.
    """
    average_accuracies = []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            try:
                accuracies = [float(value) for value in row]
                if accuracies:
                    average_accuracy = sum(accuracies) / len(accuracies)
                    average_accuracies.append(average_accuracy)
            except ValueError:
                print(f"Skipping invalid row: {row}")
    return average_accuracies

def main():
    parser = argparse.ArgumentParser(description='Compute weights based on average accuracy for ensemble learning.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file containing accuracies')
    args = parser.parse_args()
    
    average_accuracies = read_accuracies_from_csv(args.input_file)
    
    if not average_accuracies:
        print("No valid accuracies found in the input file.")
        return
    
    accuracy_based_weights = compute_accuracy_based_weights(average_accuracies)

    # Print weights rounded to two decimal places
    formatted_weights = [round(w, 2) for w in accuracy_based_weights]
    print("Accuracy-based weights:", formatted_weights)
    print("Sum of weights:", round(sum(accuracy_based_weights), 2))  # Should equal 1.00

if __name__ == "__main__":
    main()
