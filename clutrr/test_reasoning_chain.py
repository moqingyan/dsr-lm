import os
import csv

from argparse import ArgumentParser
from tqdm import tqdm

import scallopy

relation_id_map = {
  'daughter': 0,
  'sister': 1,
  'son': 2,
  'aunt': 3,
  'father': 4,
  'husband': 5,
  'granddaughter': 6,
  'brother': 7,
  'nephew': 8,
  'mother': 9,
  'uncle': 10,
  'grandfather': 11,
  'wife': 12,
  'grandmother': 13,
  'niece': 14,
  'grandson': 15,
  'son-in-law': 16,
  'father-in-law': 17,
  'daughter-in-law': 18,
  'mother-in-law': 19,
}

class ReasoningChainDataset:
  def __init__(self, file_names):
    self.data = []
    for file_name in file_names:
      file_csv_content = csv.reader(open(file_name))
      for row in list(file_csv_content)[1:]:
        self.data.append((row[-9].split("-"), row[5]))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    (chain, result) = self.data[i]
    context_facts = [(relation_id_map[r], f"{j}", f"{j + 1}") for (j, r) in enumerate(chain)]
    question_facts = [(f"0", f"{len(chain)}")]
    facts = {"context": context_facts, "question": question_facts}
    result_id = relation_id_map[result]
    return (facts, result_id)


class ScallopReasoning:
  def __init__(self):
    self.ctx = scallopy.ScallopContext()
    self.ctx.import_file(os.path.abspath(os.path.join(os.path.abspath(__file__), "../scl/clutrr.scl")))

  def test_datapoint(self, datapoint):
    (facts, gt) = datapoint
    ctx = self.ctx.clone()
    ctx.add_facts("context", facts["context"])
    ctx.add_facts("question", facts["question"])
    ctx.run()
    pred = list(ctx.relation("answer"))
    return gt in [r for (r,) in pred]


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--dataset", type=str, default="data_089907f8")
  args = parser.parse_args()

  # Load files
  dataset_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f"../../data/CLUTRR/{args.dataset}/"))
  file_names = [os.path.join(dataset_dir, d) for d in os.listdir(dataset_dir) if ".csv" in d]

  # Load proof states
  dataset = ReasoningChainDataset(file_names)
  reasoning = ScallopReasoning()
  iterator = tqdm(range(len(dataset)))
  num_correct = 0
  for i in iterator:
    datapoint = dataset[i]
    correct = reasoning.test_datapoint(datapoint)
    if not correct:
      print(datapoint)
    num_correct += 1 if correct else 0
    accuracy = float(num_correct) / float(i + 1)
    iterator.set_description(f"Accuracy {accuracy:.4f}% ({num_correct}/{i + 1})")
