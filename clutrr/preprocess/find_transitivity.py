import os
import csv
import json

from argparse import ArgumentParser

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--dataset", type=str, default="data_089907f8")
  args = parser.parse_args()

  # Load files
  dataset_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f"../../../data/CLUTRR/{args.dataset}/"))
  file_names = [os.path.join(dataset_dir, d) for d in os.listdir(dataset_dir) if ".csv" in d]

  # Load proof states
  all_transitivities = set()
  for file_name in file_names:
    file_csv_content = csv.reader(open(file_name))
    for row in list(file_csv_content)[1:]:
      print(row[-9], row[5])
