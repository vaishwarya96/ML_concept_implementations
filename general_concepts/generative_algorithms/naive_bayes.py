from csv import reader
import math


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

file_name = 'iris_data.csv'
dataset = load_csv(file_name)
print(len(dataset))