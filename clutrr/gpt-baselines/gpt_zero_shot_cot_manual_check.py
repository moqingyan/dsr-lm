import json
import os

from tqdm import tqdm
from argparse import ArgumentParser

def check_correctness_of_prediction(directory, correctness_record, progressbar, iterator):
  ls = os.listdir(directory)
  for f in ls:
    if "response_strings.json" in f:
      d = os.path.join(directory, f)
      if not check_correctness_of_file(d, correctness_record, progressbar, iterator):
        return

def check_correctness_of_file(filename, correctness_record, progressbar, iterator):
  response = json.load(open(filename))
  for datapoint in response:
    task = datapoint[0][0]
    uuid = datapoint[0][1]
    answer = datapoint[3]
    completion = datapoint[1].strip()

    x = next(iterator)
    if x < 827:
      continue

    if answer not in completion: # or datapoint[2][0] not in completion or datapoint[2][1] not in completion:
      correctness_record[uuid] = [datapoint[0][0], False]
    else:
      print(f"Metadata: [{filename}] [{datapoint[0][0]}] [{datapoint[0][1]}]")
      print(f"GPT-3 Completion: {completion}")
      print(f"Query: ({datapoint[2][0]}, {datapoint[2][1]})")
      print(f"Answer: {answer}")
      result = input("Is this correct? [y/n/s] ")

      # Early termination
      if result == "s":
        return False

      correctness_record[uuid] = [task, result == "y"]

      # Store temporarily
      json.dump(correctness_record, open(os.path.join(args.directory, "correctness_record.json"), "w"))

    # Compute accuracy
    num_correct = len([() for v in correctness_record.values() if v[1]])
    total_num = len(correctness_record)
    accuracy = float(num_correct) / float(total_num)
    progressbar.set_description(f"[{num_correct}/{total_num} ({accuracy:.4f})]")

  return True

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--directory", default="experiments/model/clutrr/gpt_preds_zero_cot_pred_all_1")
  args = parser.parse_args()

  progressbar = tqdm(range(72 * 16))
  iterator = progressbar.__iter__()
  correctness_record = {}
  check_correctness_of_prediction(args.directory, correctness_record, progressbar, iterator)

  num_correct = len([() for v in correctness_record.values() if v[1]])
  total_num = len(correctness_record)
  accuracy = float(num_correct) / float(total_num)
  print(f"Final Assessment: [{num_correct}/{total_num} ({accuracy:.4f})]")

  for i in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    num_correct = len([() for v in correctness_record.values() if f"1.{i}" in v[0] and v[1]])
    total_num = len([() for v in correctness_record.values() if f"1.{i}" in v[0]])
    if total_num != 0:
      accuracy = float(num_correct) / float(total_num)
      print(f"Accuracy for task_1.{i}: [{num_correct}/{total_num} ({accuracy:.4f})]")

  json.dump(correctness_record, open(os.path.join(args.directory, "correctness_record.json"), "w"))
