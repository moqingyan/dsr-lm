import os
import csv
import re

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

class NamePairDataset:
  def __init__(self, file_names):
    self.data = []
    for file_name in file_names:
      file_csv_content = csv.reader(open(file_name))
      for (i, row) in enumerate(list(file_csv_content)[1:]):
        context = row[2]
        proof_state = row[8]
        self.data.append((file_name, i, context, proof_state))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    return self.data[i]

def preprocess_contexts(contexts, context_splits):
  clean_context_splits = []
  clean_contexts = []
  name_token_indices_maps = []
  for (_, (start, end)) in enumerate(context_splits):
    skip_next = False
    skip_until = 0
    curr_clean_contexts = []
    curr_name_token_indices_maps = []
    for (j, sentence) in zip(range(start, end), contexts[start:end]):
      # It is possible to skip a sentence because the previous one includes the current one.
      if skip_next:
        if j >= skip_until:
          skip_next = False
        continue

      # Get all the names of the current sentence
      names = re.findall("\\[(\w+)\\]", sentence)

      # Check if we need to include the next sentence(s) as well
      num_sentences = 1
      union_sentence = f"{sentence}"
      for k in range(j + 1, end):
        next_sentence = contexts[k]
        next_sentence_names = re.findall("\\[(\w+)\\]", next_sentence)
        if len(names) == 1 or len(next_sentence_names) == 1:
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
      names = set(names)

      # Debug number of sentences
      if args.debug and num_sentences > 1:
        print(f"number of sentences: {num_sentences}, number of names: {len(names)}; {names}")
        print("Sentence:", union_sentence)

      # Then split the context by `[` and `]` so that names are isolated in its own string
      splitted = [u.strip() for t in union_sentence.split("[") for u in t.split("]") if u.strip() != ""]

      # Get the ids of the name in the `splitted` array
      is_name_ids = {s: [j for (j, sp) in enumerate(splitted) if sp == s] for s in names}

      # Clean up the sentence and add it to the batch
      clean_sentence = union_sentence.replace("[", "").replace("]", "")

      # Preprocess the context
      curr_clean_contexts.append(clean_sentence)
      curr_name_token_indices_maps.append(is_name_ids)

    # Add this batch into the overall list; record the splits
    curr_size = len(curr_clean_contexts)
    clean_context_splits.append((0, curr_size) if len(clean_context_splits) == 0 else (clean_context_splits[-1][1], clean_context_splits[-1][1] + curr_size))
    clean_contexts += curr_clean_contexts
    name_token_indices_maps += curr_name_token_indices_maps

  # Return the preprocessed contexts and splits
  return (clean_contexts, clean_context_splits, name_token_indices_maps)

def extract_name_pairs_from_context(context):
  contexts = [s.strip().lower() for s in context.split(".") if s.strip() != ""]
  splits = [(0, len(contexts))]
  (clean_contexts, _, name_token_indices_maps) = preprocess_contexts(contexts, splits)
  name_pairs = []
  for i in range(len(clean_contexts)):
    names = list(name_token_indices_maps[i].keys())
    name_pairs += [(m, n) for m in names for n in names if m != n]
  return set(name_pairs)

def extract_base_name_pairs_from_proof_state(proof_state):
  proof_state = eval(proof_state)
  proven = set()
  for i in proof_state:
    for (sub, _, obj) in i.keys():
      pair = (sub.lower(), obj.lower())
      proven.add(pair)
  base = set()
  for i in proof_state:
    for (_, facts) in i.items():
      for (sub, _, obj) in facts:
        pair = (sub.lower(), obj.lower())
        if pair not in proven:
          base.add(pair)
  return base


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--dataset", type=str, default="data_089907f8")
  parser.add_argument("--debug", action="store_true")
  args = parser.parse_args()

  # Load files
  dataset_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f"../../data/CLUTRR/{args.dataset}/"))
  file_names = [os.path.join(dataset_dir, d) for d in os.listdir(dataset_dir) if ".csv" in d and "1.2" in d]

  # Load proof states
  dataset = NamePairDataset(file_names)
  iterator = tqdm(range(len(dataset)))
  num_correct = 0
  for i in iterator:
    (file_name, j, contexts, proof_state) = dataset[i]
    context_name_pairs = extract_name_pairs_from_context(contexts)
    gt_name_pairs = extract_base_name_pairs_from_proof_state(proof_state)
    correct = gt_name_pairs.issubset(context_name_pairs)
    if not correct:
      print(file_name, j)
      print(contexts)
      print("Extracted:", context_name_pairs)
      print("Required:", gt_name_pairs)
      print("Missing:", gt_name_pairs.difference(context_name_pairs))
    num_correct += 1 if correct else 0
    accuracy = float(num_correct) / float(i + 1)
    iterator.set_description(f"Accuracy {accuracy:.4f}% ({num_correct}/{i + 1})")
