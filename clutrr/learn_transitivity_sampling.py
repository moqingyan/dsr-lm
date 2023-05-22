import os

from argparse import ArgumentParser
from tqdm import tqdm
import csv
import re
import random
import transformers

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
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
  'nothing': 20,
}

id_relation_map = { val: key for key, val in relation_id_map.items()}

all_possible_transitives = [(a, b, c) for a in range(20) for b in range(20) for c in range(20) ]

class CLUTRRDataset:
  def __init__(self, root, dataset, split):
    self.dataset_dir = os.path.join(root, f"{dataset}")
    self.file_names = [os.path.join(self.dataset_dir, d) for d in os.listdir(self.dataset_dir) if f"_{split}.csv" in d]
    self.data = [row for f in self.file_names for row in list(csv.reader(open(f)))[1:]]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    # Context is a list of sentences
    # context = [s.strip().lower() for s in self.data[i][2].split(".") if s.strip() != ""]
    gt_relations = eval(self.data[i][8])
    deduced_relations = set([k for i in gt_relations for k in i.keys()])
    given_relations = set([t for i in gt_relations for v in i.values() for t in v])
    leave_relations = given_relations.difference(deduced_relations)
    leave_relations = [(relation_id_map[r], s.lower(), o.lower()) for (s, r, o) in leave_relations]

    # Query is of type (sub, obj)
    query_sub_obj = eval(self.data[i][3])
    query = (query_sub_obj[0].lower(), query_sub_obj[1].lower())

    # Answer is one of 20 classes such as daughter, mother, ...
    answer = self.data[i][5]
    return ((leave_relations, query), answer)

  @staticmethod
  def collate_fn(batch):
    queries = [[query] for ((_, query), _) in batch]
    batched_gt_relations = [gt_relations for ((gt_relations, _), _) in batch]
    answers = torch.stack([torch.tensor(relation_id_map[answer]) for (_, answer) in batch])
    return ((batched_gt_relations, queries), answers)

def clutrr_loader(root, dataset, batch_size):
  train_dataset = CLUTRRDataset(root, dataset, "train")
  train_loader = DataLoader(train_dataset, batch_size, collate_fn=CLUTRRDataset.collate_fn, shuffle=True)
  test_dataset = CLUTRRDataset(root, dataset, "test")
  test_loader = DataLoader(test_dataset, batch_size, collate_fn=CLUTRRDataset.collate_fn, shuffle=True)
  return (train_loader, test_loader)


class CLUTRRModel(nn.Module):
  def __init__(
    self,
    device="cpu",
    debug=False,
    no_fine_tune_roberta=False,
    provenance="difftopbottomkclauses",
    train_top_k=3,
    test_top_k=3,
    scallop_softmax=False,
    sample_ct = 150,
  ):
    super(CLUTRRModel, self).__init__()

    # Options
    self.device = device
    self.debug = debug
    self.no_fine_tune_roberta = no_fine_tune_roberta
    self.scallop_softmax = scallop_softmax
    self.sample_ct = sample_ct

    # Transitivity probs: Initialize with 0.1
    self.transitivity_probs = torch.tensor(np.ones(len(all_possible_transitives))/10, requires_grad=True, device=self.device)

    # Scallop reasoning context
    self.scallop_ctx = scallopy.ScallopContext(provenance=provenance, train_k=train_top_k, test_k=test_top_k)
    self.scallop_ctx.import_file(os.path.abspath(os.path.join(os.path.abspath(__file__), "../scl/clutrr_no_transitivity.scl")))
    self.scallop_ctx.set_iter_limit(10)
    self.scallop_ctx.set_non_probabilistic(["question"])
    self.scallop_ctx.set_non_probabilistic(["context"])

    if self.debug:
      self.reason = self.scallop_ctx.forward_function("answer", list(range(len(relation_id_map))), dispatch="single", debug_provenance=True)
    else:
      self.reason = self.scallop_ctx.forward_function("answer", list(range(len(relation_id_map))))

  def forward(self, x, phase='train'):
    (relations, queries) = x

    # Debug prints
    if self.debug:
      print(relations)
      print(queries)

    transitivity_probs = torch.clamp(self.transitivity_probs, 0, 1)
    # orig_transitive_relations = [[(prob, relation) for prob, relation in zip(self.transitivity_probs, all_possible_transitives)] for _ in range(len(relations))]

    if phase == 'train':
      sampled_transitive_idx = torch.multinomial(transitivity_probs, self.sample_ct)
    else:
      _, sampled_transitive_idx = torch.topk(transitivity_probs, self.sample_ct)

    probs = transitivity_probs[sampled_transitive_idx]
    transitives = [all_possible_transitives[i] for i in sampled_transitive_idx]
    transitive_relations = [[(prob, relation) for prob, relation in zip(probs, transitives)] for _ in range(len(relations))]

    # Go though the preprocessing, RoBERTa model forwarding, and facts extraction steps
    # (splits, name_pairs, name_pair_relations) = self._extract_relations(clean_contexts, clean_context_splits, name_token_indices_maps)
    # (context_facts, context_disjunctions, question_facts) = self._extract_facts(splits, name_pairs, name_pair_relations, queries)

    # Run Scallop to reason the result relation
    result = self.reason(context=relations, transitive=transitive_relations, question=queries)

    if self.scallop_softmax:
      result = torch.softmax(result, dim=1)
    result = torch.clamp(result, 0, 1)

    # Return the final result
    return result


class Trainer:
  def __init__(self, train_loader, test_loader, device, model_dir, model_name, learning_rate, **args):
    self.device = device
    load_model = args.pop('load_model')
    if load_model:
      self.model = torch.load(os.path.join(model_dir, model_name + '.best.model'))
    else:
      self.model = CLUTRRModel(device=device, **args).to(device)
    self.model_dir = model_dir
    self.model_name = model_name
    self.optimizer = optim.Adam([self.model.transitivity_probs], lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.min_test_loss = 10000000000.0
    self.max_accu = 0

  def loss(self, y_pred, y):
    (_, dim) = y_pred.shape
    gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)], device=self.device) for t in y])
    return nn.functional.binary_cross_entropy(y_pred, gt)

  def accuracy(self, y_pred, y):
    batch_size = len(y)
    pred = torch.argmax(y_pred, dim=1)
    num_correct = len([() for i, j in zip(pred, y) if i == j])
    return (num_correct, batch_size)

  def train(self, num_epochs):
    # self.test_epoch(0)
    for i in range(1, num_epochs + 1):
      self.train_epoch(i)
      self.test_epoch(i)

  def train_epoch(self, epoch):
    self.model.train()
    total_count = 0
    total_correct = 0
    total_loss = 0
    iterator = tqdm(self.train_loader)
    for (i, (x, y)) in enumerate(iterator):
      self.optimizer.zero_grad()
      y_pred = self.model(x, 'train')
      loss = self.loss(y_pred, y)
      total_loss += loss.item()
      loss.backward()
      self.optimizer.step()

      (num_correct, batch_size) = self.accuracy(y_pred, y)
      total_count += batch_size
      total_correct += num_correct
      correct_perc = 100. * total_correct / total_count
      avg_loss = total_loss / (i + 1)

      iterator.set_description(f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Accuracy: {total_correct}/{total_count} ({correct_perc:.2f}%)")

  def test_epoch(self, epoch):
    self.model.eval()
    total_count = 0
    total_correct = 0
    total_loss = 0
    with torch.no_grad():
      iterator = tqdm(self.test_loader)
      for (i, (x, y)) in enumerate(iterator):
        y_pred = self.model(x, 'test')
        loss = self.loss(y_pred, y)
        total_loss += loss.item()

        (num_correct, batch_size) = self.accuracy(y_pred, y)
        total_count += batch_size
        total_correct += num_correct
        correct_perc = 100. * total_correct / total_count
        avg_loss = total_loss / (i + 1)

        iterator.set_description(f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Accuracy: {total_correct}/{total_count} ({correct_perc:.2f}%)")

    # Save model
    if total_correct / total_count > self.max_accu:
      self.max_accu = total_correct / total_count
      torch.save(self.model, os.path.join(self.model_dir, f"{self.model_name}.best.model"))
    torch.save(self.model, os.path.join(self.model_dir, f"{self.model_name}.latest.model"))

  def get_rules(self, threshold=0.6):
    pred_probs = self.model.transitivity_probs.reshape(-1)
    rules = pred_probs > threshold
    indices = rules.nonzero()
    rules = sorted([(pred_probs[index].item(), [id_relation_map[e] for e in all_possible_transitives[index]]) for index in indices], reverse=True)
    return rules

def pretty_print_rules(rules):
  rule_str = '\n'.join([f"{p};{r}" for p, r in rules])
  print(rule_str)

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--dataset", type=str, default="data_089907f8")
  parser.add_argument("--model-name", type=str, default="learn_transitivity_sample")
  parser.add_argument("--load_model", type=bool, default=False)
  parser.add_argument("--n-epochs", type=int, default=1000)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--seed", type=int, default=5432)
  parser.add_argument("--learning-rate", type=float, default=0.01)
  parser.add_argument("--num-mlp-layers", type=int, default=2)
  parser.add_argument("--sample_ct", type=int, default=100)
  parser.add_argument("--provenance", type=str, default="difftopbottomkclauses")
  parser.add_argument("--train-top-k", type=int, default=3)
  parser.add_argument("--test-top-k", type=int, default=3)
  parser.add_argument("--no-fine-tune-roberta", type=bool, default=False)
  parser.add_argument("--use-softmax", type=bool, default=True)
  parser.add_argument("--scallop-softmax", type=bool, default=False)
  parser.add_argument("--debug", type=bool, default=False)
  parser.add_argument("--use-last-hidden-state", action="store_true")
  parser.add_argument("--cuda", type=bool, default=True)
  parser.add_argument("--gpu", type=int, default=0)
  args = parser.parse_args()
  print(args)

  args.model_name = f"{args.model_name}_{args.sample_ct}"
  args.model_name = "learn_transitivity_init_0.5"
  # Parameters: for some reason, the seed cannot fully control the reproductivity
  # Perhaps due to some error in pytorch c binding?
  # The problem occurs for more than 2000 datapoints
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)
  random.seed(args.seed)
  np.random.seed(args.seed)
  transformers.set_seed(args.seed)
  # transformers.enable_full_determinism(args.seed)

  if args.cuda:
    if torch.cuda.is_available(): device = torch.device(f"cuda:{args.gpu}")
    else: raise Exception("No cuda available")
  else: device = torch.device("cpu")

  # Setting up data and model directories
  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/clutrr"))
  if not os.path.exists(model_dir): os.makedirs(model_dir)

  # Load the dataset
  (train_loader, test_loader) = clutrr_loader(data_root, args.dataset, args.batch_size)

  # Train
  trainer = Trainer(train_loader, test_loader, device, model_dir, args.model_name, args.learning_rate, debug=args.debug, provenance=args.provenance, train_top_k=args.train_top_k, test_top_k=args.test_top_k, no_fine_tune_roberta=args.no_fine_tune_roberta, scallop_softmax = args.scallop_softmax, load_model=args.load_model, sample_ct=args.sample_ct)
  trainer.train(args.n_epochs)
  # trainer.test_epoch(0)
  # rules = trainer.get_rules()
  # pretty_print_rules(rules)
