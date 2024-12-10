import csv
import sys

if len(sys.argv) != 3:
    print("Usage: python script.py input_file.csv output_file.csv")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # Skip the first row
    next(reader, None)
    
    for row in reader:
        # Exclude the first and last columns
        modified_row = row[1:-1]
        writer.writerow(modified_row)

print(f"Modified CSV file has been saved to '{output_file}'.")
