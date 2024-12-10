# IMPORTS (for Gurobi, CSV, SYS)
import csv,sys
import argparse
import gurobipy as gp
import numpy as np
from gurobipy import GRB

def count_rows(file_path):
    """
    Count the number of rows in the given CSV file.

    Parameters:
    - file_path (str): Path to the input CSV file.

    Returns:
    - row_count (int): Number of rows in the file.
    """
    row_count = 0
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            row_count += 1
    return row_count

def count_columns(file_path):
    """
    Count the number of columns in the given CSV file.

    Parameters:
    - file_path (str): Path to the input CSV file.

    Returns:
    - column_count (int): Number of columns in the file.
    """
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            column_count = len(row)
            break  # Assuming all rows have the same number of columns
    return column_count

def gurobi(number_rows, number_columns, input_file_name, result_file):
    #   Read the input for the optimisation problem 
    rows, cols = (number_rows, number_columns)
    acc = [[0] * cols] * rows
    #   INPUT ACCURACY MATRIX FROM CSV
    CSVData = open(input_file_name)
    #   ACCURACY CSVDATA
    acc = np.loadtxt(CSVData, delimiter=",")
    algorithms = number_rows
    classes = number_columns
    #   Generate Gurobi Model (m) and the variables to be considered in the solver
    m = gp.Model("mip")
    
    # CONSTRAINT 6
    x = m.addVars(algorithms, vtype=GRB.BINARY, name="x")

    # CONSTRAINT 7
    w = m.addVars(algorithms, classes, vtype=GRB.CONTINUOUS, name="w")
    for j in range(0, classes):
        for i in range(0, algorithms):
            m.addConstr(w[i, j] >= 0)

    # CONSTRAINT 8,9
    K = m.addVar(vtype=GRB.INTEGER, name="KAPPA")
    m.addConstr(K == 8)
    #m.addConstr(K <= algorithms)

    # CONSTRAINT 10
    for i in range(0, algorithms):
        sum_of_weights_per_class = 0.0
        for j in range(0, classes):
            sum_of_weights_per_class += (w[i, j])
        m.addConstr(sum_of_weights_per_class <= (classes * x[i]))

    # CONSTRAINT 11
    M = 2
    for i in range(0, algorithms):
        sum_of_weights_per_class_and_M = 0.0
        for j in range(0, classes):
            sum_of_weights_per_class_and_M += (w[i, j])
        sum_of_weights_per_class_and_M += (M * (1 - x[i]))
        m.addConstr(sum_of_weights_per_class_and_M >= 0.0001)

    # CONSTRAINT 12
    for j in range(0, classes):
        sum_of_weighs_per_algorithm = 0.0
        for i in range(0, algorithms):
            sum_of_weighs_per_algorithm += (w[i, j])
        m.addConstr(sum_of_weighs_per_algorithm == 1)

    # CONSTRAINT 13
    sum_of_selection_x = 0.0
    for i in range(0, algorithms):
        sum_of_selection_x += x[i]
    m.addConstr(K == sum_of_selection_x)

    # CONSTRAINT 14
    for j in range(0, classes):
        algorithm_weighted_class_average = 0.0
        algorithm_class_averaged = 0.0
        for i in range(0, algorithms):
            algorithm_weighted_class_average += (w[i, j] * acc[i, j])
            algorithm_class_averaged += (acc[i, j])
        algorithm_class_averaged = algorithm_class_averaged / algorithms
        m.addConstr(algorithm_weighted_class_average >= algorithm_class_averaged)

    # CONSTRAINT 15 (EXTRA COMPUTATION)
    weighted_accuracy_sum = 0.0
    for i in range(0, algorithms):
        for j in range(0, classes):
            weighted_accuracy_sum += (w[i, j] * acc[i, j])
    weighted_accuracy_sum_divided = weighted_accuracy_sum / classes

    accuracy_sum = 0.0
    for i in range(0, algorithms):
        for j in range(0, classes):
            accuracy_sum += acc[i, j]
    averaged_accuracy_sum = accuracy_sum / (classes * algorithms)
    m.addConstr(weighted_accuracy_sum_divided >= averaged_accuracy_sum)
    
    # New version with regularization
    valueAlpha = 0.82
    valueLambda = 1.00
    
    weighted_accuracy_squared = 0.0
    for i in range(0, algorithms):
        for j in range(0, classes):
            weighted_accuracy_squared += (w[i, j] * w[i, j])
    
    weighted_accuracy_single = 0.0
    for i in range(0, algorithms):
        for j in range(0, classes):
            weighted_accuracy_single += w[i, j]
            
    parenthesisW = (((1-valueAlpha) * 0.5) * weighted_accuracy_squared) + (valueAlpha * weighted_accuracy_single)
    llambdaparenthesisW = valueLambda * (parenthesisW)
    
    # UPDATED
    f1 = weighted_accuracy_sum_divided - llambdaparenthesisW
    # OBJECTIVE
    m.setObjective(f1, GRB.MAXIMIZE)
    m.optimize()
    # PRINT SOLUTION
    solutionCounter = 0
    if m.status == GRB.OPTIMAL:
        all_vars = m.getVars()
        values = m.getAttr("X", all_vars)
        names = m.getAttr("VarName", all_vars)

        for name, val in zip(names, values):
            print(f"{name} = {val:.2f}")
    else:
        print("No solution found")

def main():
    parser = argparse.ArgumentParser(description='Compute the number of rows and columns in a CSV file.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file')
    args = parser.parse_args()

    file_path = args.input_file

    rows = count_rows(file_path)
    columns = count_columns(file_path)

    print(f"Number of rows: {rows}")
    print(f"Number of columns: {columns}")
    gurobi(rows,columns,file_path,file_path+"_sol.txt")

if __name__ == "__main__":
    main()