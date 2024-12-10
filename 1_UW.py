import argparse

def calculate_uniform_weights(num_learners):
    """
    Calculate uniform weights for ensemble learning.

    Parameters:
    - num_learners (int): Number of base learners.

    Returns:
    - weights (list): List of uniform weights for each base learner.
    """
    uniform_weight = 1.0 / num_learners
    weights = [uniform_weight] * num_learners
    return weights

def count_lines_in_file(file_path):
    """
    Count the number of lines in the given file.

    Parameters:
    - file_path (str): Path to the input file.

    Returns:
    - num_lines (int): Number of lines in the file.
    """
    with open(file_path, 'r') as file:
        num_lines = sum(1 for _ in file)
    return num_lines

def main():
    parser = argparse.ArgumentParser(description='Calculate uniform weights for ensemble learning.')
    parser.add_argument('input_file', type=str, help='Path to the input file')
    args = parser.parse_args()
    
    num_learners = count_lines_in_file(args.input_file)
    uniform_weights = calculate_uniform_weights(num_learners)
    print("Uniform weights:", uniform_weights)

if __name__ == "__main__":
    main()
