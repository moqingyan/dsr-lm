import os
import csv
import random
from argparse import ArgumentParser
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from clutrr.run import relation_id_map, CLUTRRModel, MLP

class CLUTRRTestDataset:
  def __init__(self, root, dataset, lengths=[2, 3]):
    self.dataset_dir = os.path.join(root, f"CLUTRR/{dataset}/")
    self.file_names = [os.path.join(self.dataset_dir, f"1.{l}_test.csv") for l in lengths]
    self.data = [row for f in self.file_names for row in list(csv.reader(open(f)))[1:]]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    datapoint = self.data[i]

    # Context is a list of sentences
    context = [s.strip().lower() for s in datapoint[2].split(".") if s.strip() != ""]

    # Query is of type (sub, obj)
    query_sub_obj = eval(datapoint[3])
    query = (query_sub_obj[0].lower(), query_sub_obj[1].lower())

    # Answer is one of 20 classes such as daughter, mother, ...
    answer = datapoint[5]
    return ((context, query), answer)

  @staticmethod
  def collate_fn(batch):
    queries = [query for ((_, query), _) in batch]
    contexts = [fact for ((context, _), _) in batch for fact in context]
    context_lens = [len(context) for ((context, _), _) in batch]
    context_splits = [(sum(context_lens[:i]), sum(context_lens[:i + 1])) for i in range(len(context_lens))]
    answers = torch.stack([torch.tensor(relation_id_map[answer]) for (_, answer) in batch])
    return ((contexts, queries, context_splits), answers)


class Tester:
  def __init__(self, test_loader, model, detail=False):
    self.test_loader = test_loader
    self.model = model
    self.detail = detail

  def loss(self, y_pred, y):
    (_, dim) = y_pred.shape
    gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in y])
    return nn.functional.binary_cross_entropy(y_pred, gt)

  def accuracy(self, y_pred, y):
    batch_size = len(y)
    pred = torch.argmax(y_pred, dim=1)
    num_correct = len([() for i, j in zip(pred, y) if i == j])
    return (num_correct, batch_size)

  def predict_in_detail(self, x, y):
    (contexts, queries, context_splits) = x
    (clean_contexts, clean_context_splits, name_token_indices_maps) = self.model._preprocess_contexts(contexts, context_splits)
    (splits, name_pairs, name_pair_relations) = self.model._extract_relations(clean_contexts, clean_context_splits, name_token_indices_maps)
    (context_facts, context_disjunctions, question_facts) = self.model._extract_facts(splits, name_pairs, name_pair_relations, queries)
    result = self.model.reason(context=context_facts, question=question_facts, disjunctions={"context": context_disjunctions})
    return result

  def test(self):
    self.model.eval()
    total_count = 0
    total_correct = 0
    total_loss = 0
    with torch.no_grad():
      iterator = tqdm(self.test_loader)
      for (i, (x, y)) in enumerate(iterator):
        y_pred = self.predict_in_detail(x, y) if self.detail else self.model(x).to("cpu")
        loss = self.loss(y_pred, y)
        total_loss += loss.item()
        (num_correct, batch_size) = self.accuracy(y_pred, y)
        total_count += batch_size
        total_correct += num_correct
        correct_perc = 100. * total_correct / total_count
        avg_loss = total_loss / (i + 1)
        iterator.set_description(f"[Test] Avg Loss: {avg_loss}, Accuracy: {total_correct}/{total_count} ({correct_perc:.2f}%)")


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--dataset", type=str, default="data_089907f8")
  parser.add_argument("--model-name", type=str, default="clutrr_global_before_lstm.best.model")
  parser.add_argument("--lengths", nargs="+", type=int, default=[2])
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--cuda", action="store_true")
  parser.add_argument("--gpu", type=int, default=2)
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--detail", action="store_true")
  args = parser.parse_args()

  # Temporary
  args.cuda = True
  args.detail = True

  # Parameters
  torch.manual_seed(args.seed)
  random.seed(args.seed)
  if args.cuda:
    if torch.cuda.is_available(): device = torch.device(f"cuda:{args.gpu}")
    else: raise Exception("No cuda available")
  else: device = torch.device("cpu")

  # Setting up data and model directories
  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/clutrr"))

  # Setup dataset
  test_dataset = CLUTRRTestDataset(data_root, args.dataset, args.lengths)
  test_loader = DataLoader(test_dataset, args.batch_size, collate_fn=CLUTRRTestDataset.collate_fn, shuffle=True)

  # Load model
  model = torch.load(os.path.join(model_dir, args.model_name))

  # Enter test main loop
  tester = Tester(test_loader, model, detail=args.detail)
  tester.test()
