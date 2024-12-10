import argparse

def calculate_uniform_weights(num_learners):
    """
    Calculate uniform weights for ensemble learning.

    Parameters:
    - num_learners (int): Number of base learners.

    Returns:
    - weights (list): List of uniform weights for each base learner, formatted to two decimal places.
    """
    uniform_weight = 1.0 / num_learners
    weights = [round(uniform_weight, 2) for _ in range(num_learners)]
    return weights

def count_columns_in_file(file_path):
    """
    Count the total number of columns in each line of the given file.

    Parameters:
    - file_path (str): Path to the input file.

    Returns:
    - total_columns (int): Total number of columns in the file.
    """
    with open(file_path, 'r') as file:
        total_columns = 0
        for line in file:
            # Count the number of columns in the current line
            num_columns_in_line = len(line.split())
            total_columns += num_columns_in_line
    return total_columns

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
    num_classes  = count_columns_in_file(args.input_file)
    uniform_weights = calculate_uniform_weights(num_learners*num_classes)
    print("Uniform weights:", uniform_weights)

if __name__ == "__main__":
    main()
