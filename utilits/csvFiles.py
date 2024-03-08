import umap
import pandas as pd
import csv
import sys
import numpy as np

def convertToString(mapper):
   data = np.fromstring(mapper[1:-1], sep=' ')
   return

def create_csv(mapper):
 output_csv_path = "/Users/bharadwajanandivada/SCRNA_DeepLearnModel/barcodes/barcodes_updated.csv"

 with open("/Users/bharadwajanandivada/SCRNA_DeepLearnModel/barcodes/barcodes.csv", 'r') as file:
    csvreader = csv.reader(file)


    with open(output_csv_path, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=['barcodes', 'x', 'y'])
        writer.writeheader()


        for row in csvreader:

            writer.writerow({'barcodes': row[0], 'x': convertToString(mapper[0]), 'y': convertToString(mapper[1])})
            print("CSV file created successfully.")
    return 0
