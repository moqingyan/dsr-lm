import os

from argparse import ArgumentParser
from tqdm import tqdm
import csv
import re
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizer
import numpy as np

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

class CLUTRRDataset:
  def __init__(self, root, dataset, split):
    self.dataset_dir = os.path.join(root, f"{dataset}/")
    self.file_names = [os.path.join(self.dataset_dir, d) for d in os.listdir(self.dataset_dir) if f"_{split}.csv" in d]
    self.data = [row for f in self.file_names for row in list(csv.reader(open(f)))[1:]]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    # Context is a list of sentences
    context = [s.strip().lower() for s in self.data[i][2].split(".") if s.strip() != ""]

    # Query is of type (sub, obj)
    query_sub_obj = eval(self.data[i][3])
    query = (query_sub_obj[0].lower(), query_sub_obj[1].lower())

    # Answer is one of 20 classes such as daughter, mother, ...
    answer = self.data[i][5]
    return ((context, query), answer)

  @staticmethod
  def collate_fn(batch):
    queries = [query for ((_, query), _) in batch]
    contexts = [fact for ((context, _), _) in batch for fact in context]
    context_lens = [len(context) for ((context, _), _) in batch]
    context_splits = [(sum(context_lens[:i]), sum(context_lens[:i + 1])) for i in range(len(context_lens))]
    answers = torch.stack([torch.tensor(relation_id_map[answer]) for (_, answer) in batch])
    return ((contexts, queries, context_splits), answers)


def clutrr_loader(root, dataset, batch_size):
  train_dataset = CLUTRRDataset(root, dataset, "train")
  train_loader = DataLoader(train_dataset, batch_size, collate_fn=CLUTRRDataset.collate_fn, shuffle=True)
  test_dataset = CLUTRRDataset(root, dataset, "test")
  test_loader = DataLoader(test_dataset, batch_size, collate_fn=CLUTRRDataset.collate_fn, shuffle=True)
  return (train_loader, test_loader)


class MLP(nn.Module):
  def __init__(self, in_dim: int, embed_dim: int, out_dim: int, num_layers: int = 0, softmax = False, normalize = False, sigmoid = False):
    super(MLP, self).__init__()
    layers = []
    layers += [nn.Linear(in_dim, embed_dim), nn.ReLU()]
    for _ in range(num_layers):
      layers += [nn.Linear(embed_dim, embed_dim), nn.ReLU()]
    layers += [nn.Linear(embed_dim, out_dim)]
    self.model = nn.Sequential(*layers)
    self.softmax = softmax
    self.normalize = normalize
    self.sigmoid = sigmoid

  def forward(self, x):
    x = self.model(x)
    if self.softmax: x = nn.functional.softmax(x, dim=1)
    if self.normalize: x = nn.functional.normalize(x)
    if self.sigmoid: x = torch.sigmoid(x)
    return x


class CLUTRRModel(nn.Module):
  def __init__(self, device="cpu", num_mlp_layers=0, debug=False, use_last_hidden_state=False):
    super(CLUTRRModel, self).__init__()

    # Options
    self.device = device
    self.debug = debug
    self.use_last_hidden_state = use_last_hidden_state

    # Roberta as embedding extraction model
    self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", local_files_only=True, add_prefix_space=True)
    self.roberta_model = RobertaModel.from_pretrained("roberta-base")
    self.embed_dim = self.roberta_model.config.hidden_size

    # Entity embedding
    self.relation_extraction = MLP(self.embed_dim, self.embed_dim, len(relation_id_map), num_layers=num_mlp_layers, sigmoid=True)

    # Scallop reasoning context
    self.scallop_ctx = scallopy.ScallopContext("difftopbottomkclauses", k=3)
    self.scallop_ctx.import_file(os.path.abspath(os.path.join(os.path.abspath(__file__), "../scl/clutrr.scl")))
    self.scallop_ctx.set_non_probabilistic(["question"])
    if self.debug:
      self.reason = self.scallop_ctx.forward_function("answer", list(range(len(relation_id_map))), dispatch="single", debug_provenance=True)
    else:
      self.reason = self.scallop_ctx.forward_function("answer", list(range(len(relation_id_map))))

  def forward(self, x):
    (contexts, queries, context_splits) = x
    batch_size = len(context_splits)

    if self.debug:
      print(contexts)
      print(queries)

    # Preprocess sentences
    relation_splits = []
    relation_sentences = []
    relation_name_pairs = []
    for (_, (start, end)) in enumerate(context_splits):
      curr_relation_sentences = []
      curr_name_pairs = []
      skip_next = False
      skip_until = 0
      for (j, sentence) in zip(range(start, end), contexts[start:end]):
        # It is possible to skip a sentence because the previous one includes the current one.
        if skip_next:
          if j >= skip_until:
            skip_next = False
          continue

        # Get all the names of the current sentence
        names = re.findall("\\[(\w+)\\]", sentence)

        # Check if we need to include the next sentence(s) as well
        num_sentences_limit = 4
        num_sentences = 1
        union_sentence = f"{sentence}"
        for k in range(j + 1, end):
          next_sentence = contexts[k]
          next_sentence_names = re.findall("\\[(\w+)\\]", next_sentence)
          if (len(names) == 1 or len(next_sentence_names) == 1) and num_sentences < num_sentences_limit:
            if len(next_sentence_names) > 0:
              num_sentences += 1
              union_sentence += f". {next_sentence}"
              names += next_sentence_names
            skip_next = True
            if len(next_sentence_names) == 1:
              skip_until = k - 1
            else:
              skip_until = k
          else:
            break

        # Deduplicate the names
        names = list(dict.fromkeys(names))

        # Debug number of sentences
        if self.debug and num_sentences > 1:
          print(f"number of sentences: {num_sentences}, number of names: {len(names)}; {names}")
          print("Sentence:", union_sentence)

        # Clean up the sentence and add it to the batch
        clean_sentence = union_sentence.replace("[", "").replace("]", "")
        curr_relation_sentences += [f"{clean_sentence}. the relation between {names[k]} and {names[l]} is?" for k in range(len(names)) for l in range(len(names)) if k != l]
        curr_name_pairs += [(k, l) for k in names for l in names if k != l]

      # Construct the current datatpoint
      curr_split = (0, len(curr_relation_sentences)) if len(relation_sentences) == 0 else (relation_splits[-1][1], relation_splits[-1][1] + len(curr_relation_sentences))
      relation_sentences += curr_relation_sentences
      relation_name_pairs += curr_name_pairs
      relation_splits.append(curr_split)

    # Embed all these sentences
    sentence_tokens = self.tokenizer(relation_sentences, padding=True, return_tensors="pt")
    sentence_input_ids = sentence_tokens["input_ids"].to(self.device)
    sentence_attention_mask = sentence_tokens["attention_mask"].to(self.device)
    encoded_sentence = self.roberta_model(sentence_input_ids, sentence_attention_mask)
    if self.use_last_hidden_state:
      sentence_embeddings = encoded_sentence.last_hidden_state[:, 0, :]
    else:
      sentence_embeddings = encoded_sentence.pooler_output
    relations = self.relation_extraction(sentence_embeddings)

    # Construct facts
    question_facts = [[] for _ in range(batch_size)]
    context_facts = [[] for _ in range(batch_size)]
    for (i, (start, end)) in enumerate(relation_splits):
      question_facts[i] = [queries[i]]
      context_facts[i] = [(relations[j, k], (k, relation_name_pairs[j][0], relation_name_pairs[j][1])) for j in range(start, end) for k in range(len(relation_id_map))]

    # Run scallop
    result = self.reason(question=question_facts, context=context_facts)

    # Softmax the result
    result = nn.functional.softmax(result, dim=1)
    return result


class Trainer:
  def __init__(self, train_loader, test_loader, device, learning_rate, **args):
    self.device = device
    self.model = CLUTRRModel(device=device, **args).to(device)
    self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader

  def loss(self, y_pred, y):
    (_, dim) = y_pred.shape
    gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in y])
    return nn.functional.binary_cross_entropy(y_pred, gt)

  def accuracy(self, y_pred, y):
    batch_size = len(y)
    pred = torch.argmax(y_pred, dim=1)
    num_correct = len([() for i, j in zip(pred, y) if i == j])
    return (num_correct, batch_size)

  def train(self, num_epochs):
    for i in range(num_epochs):
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
      y_pred = self.model(x).to("cpu")
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
        y_pred = self.model(x).to("cpu")
        loss = self.loss(y_pred, y)
        total_loss += loss.item()

        (num_correct, batch_size) = self.accuracy(y_pred, y)
        total_count += batch_size
        total_correct += num_correct
        correct_perc = 100. * total_correct / total_count
        avg_loss = total_loss / (i + 1)

        iterator.set_description(f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Accuracy: {total_correct}/{total_count} ({correct_perc:.2f}%)")


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--dataset", type=str, default="data_089907f8")
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--learning-rate", type=float, default=0.00001)
  parser.add_argument("--num-mlp-layers", type=int, default=1)
  parser.add_argument("--debug", action="store_true")
  parser.add_argument("--use-last-hidden-state", action="store_true")
  parser.add_argument("--cuda", action="store_true")
  parser.add_argument("--gpu", type=int, default=0)
  args = parser.parse_args()

  # Parameters
  torch.manual_seed(args.seed)
  random.seed(args.seed)
  if args.cuda:
    if torch.cuda.is_available(): device = torch.device(f"cuda:{args.gpu}")
    else: raise Exception("No cuda available")
  else: device = torch.device("cpu")

  # Loading dataset
  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  (train_loader, test_loader) = clutrr_loader(data_root, args.dataset, args.batch_size)

  # Train
  trainer = Trainer(train_loader, test_loader, device, args.learning_rate, num_mlp_layers=args.num_mlp_layers, debug=args.debug, use_last_hidden_state=args.use_last_hidden_state)
  trainer.train(args.n_epochs)
