import os

from argparse import ArgumentParser
from tqdm import tqdm
import csv
import re
import random
import transformers
from itertools import chain
import json

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizer
from torch.optim.lr_scheduler import ExponentialLR
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
  test_loader = DataLoader(test_dataset, batch_size, collate_fn=CLUTRRDataset.collate_fn, shuffle=False)
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
  def __init__(
    self,
    gpt_pred_path,
    device="cpu",
    num_mlp_layers=1,
    debug=False,
    no_fine_tune_roberta=False,
    use_softmax=False,
    provenance="difftopbottomkclauses",
    train_top_k=3,
    test_top_k=3,
    scallop_softmax=False,
    sample_ct=150,
    prob_decay = 0.001
  ):
    super(CLUTRRModel, self).__init__()

    # Options
    self.device = device
    self.debug = debug
    self.no_fine_tune_roberta = no_fine_tune_roberta
    self.scallop_softmax = scallop_softmax
    self.sample_ct = sample_ct

    # Roberta as embedding extraction model
    self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", local_files_only=True, add_prefix_space=True)
    self.roberta_model = RobertaModel.from_pretrained("roberta-base")
    self.embed_dim = self.roberta_model.config.hidden_size

    # Entity embedding
    self.relation_extraction = MLP(self.embed_dim * 3, self.embed_dim, len(relation_id_map), num_layers=num_mlp_layers, sigmoid=not use_softmax, softmax=use_softmax)
    with open(gpt_pred_path, 'r') as f:
      gpt_preds = json.load(f)
    self.transitivity_probs = self.init_transitivity_from_file(gpt_preds)
    self.transitivity_decay = prob_decay

    # Scallop reasoning context
    self.scallop_ctx = scallopy.ScallopContext(provenance=provenance, train_k=train_top_k, test_k=test_top_k)
    self.scallop_ctx.set_iter_limit(12)
    self.scallop_ctx.import_file(os.path.abspath(os.path.join(os.path.abspath(__file__), "../scl/clutrr_ic_general.scl")))
    self.scallop_ctx.set_non_probabilistic(["question"])

    if self.debug:
      self.reason = self.scallop_ctx.forward_function(output_mappings={"answer": list(range(len(relation_id_map))), "violate_ic": [False, True]}, dispatch="single", debug_provenance=True, retain_graph=True)
    else:
      self.reason = self.scallop_ctx.forward_function(output_mappings={"answer": list(range(len(relation_id_map))), "violate_ic": [False, True]}, retain_graph=True)

  def update_decay(self):
    self.transitivity_probs = self.transitivity_probs * self.transitivity_decay

  def init_transitivity_from_file(self, gpt_preds, true_likelihood = 0.2, false_likelihood = 0.01):
    gpt_pred_ids = []
    for rel1, rel1_info in gpt_preds.items():
      for rel2, rel3 in rel1_info.items():
        if rel1 in relation_id_map and rel2 in relation_id_map and rel3 in relation_id_map:
          gpt_pred_ids.append((relation_id_map[rel1], relation_id_map[rel2], relation_id_map[rel3]))

    true_pos_ls = [all_possible_transitives.index(i) for i in gpt_pred_ids]
    all_pos_prob = np.full((len(all_possible_transitives)), false_likelihood)
    for tpos in true_pos_ls:
      all_pos_prob[tpos] = true_likelihood
    init_transitivity = torch.tensor(all_pos_prob, requires_grad=True, device=self.device)
    return init_transitivity

  def _preprocess_contexts(self, contexts, context_splits):
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
        if self.debug and num_sentences > 1:
          print(f"number of sentences: {num_sentences}, number of names: {len(names)}; {names}")
          print("Sentence:", union_sentence)

        # Then split the context by `[` and `]` so that names are isolated in its own string
        splitted = [u.strip() for t in union_sentence.split("[") for u in t.split("]") if u.strip() != ""]

        # Get the ids of the name in the `splitted` array
        is_name_ids = {s: [j for (j, sp) in enumerate(splitted) if sp == s] for s in names}

        # Get the splitted input_ids
        splitted_input_ids_raw = self.tokenizer(splitted).input_ids
        splitted_input_ids = [ids[:-1] if j == 0 else ids[1:] if j == len(splitted_input_ids_raw) - 1 else ids[1:-1] for (j, ids) in enumerate(splitted_input_ids_raw)]
        index_counter = 0
        splitted_input_indices = []
        for (j, l) in enumerate(splitted_input_ids):
          begin_offset = 1 if j == 0 else 0
          end_offset = 1 if j == len(splitted_input_ids) - 1 else 0
          quote_s_offset = 1 if "'s" in splitted[j] and splitted[j].index("'s") == 0 else 0
          splitted_input_indices.append(list(range(index_counter + begin_offset, index_counter + len(l) - end_offset - quote_s_offset)))
          index_counter += len(l) - quote_s_offset

        # Get the token indices for each name
        name_token_indices = {s: [k for phrase_id in is_name_ids[s] for k in splitted_input_indices[phrase_id]] for s in names}

        # Clean up the sentence and add it to the batch
        clean_sentence = union_sentence.replace("[", "").replace("]", "")

        # Preprocess the context
        curr_clean_contexts.append(clean_sentence)
        curr_name_token_indices_maps.append(name_token_indices)

      # Add this batch into the overall list; record the splits
      curr_size = len(curr_clean_contexts)
      clean_context_splits.append((0, curr_size) if len(clean_context_splits) == 0 else (clean_context_splits[-1][1], clean_context_splits[-1][1] + curr_size))
      clean_contexts += curr_clean_contexts
      name_token_indices_maps += curr_name_token_indices_maps

    # Return the preprocessed contexts and splits
    return (clean_contexts, clean_context_splits, name_token_indices_maps)

  def _extract_relations(self, clean_contexts, clean_context_splits, name_token_indices_maps):
    # Use RoBERTa to encode the contexts into overall tensors
    context_tokenized_result = self.tokenizer(clean_contexts, padding=True, return_tensors="pt")
    context_input_ids = context_tokenized_result.input_ids.to(self.device)
    context_attention_mask = context_tokenized_result.attention_mask.to(self.device)
    encoded_contexts = self.roberta_model(context_input_ids, context_attention_mask)
    if self.no_fine_tune_roberta:
      roberta_embedding = encoded_contexts.last_hidden_state.detach()
    else:
      roberta_embedding = encoded_contexts.last_hidden_state

    # Extract features corresponding to the names for each context
    splits, name_pairs, name_pairs_features = [], [], []

    for (begin, end) in clean_context_splits:
      curr_datapoint_name_pairs = []
      curr_datapoint_name_pairs_features = []
      curr_sentence_rep = []

      for (j, name_token_indices) in zip(range(begin, end), name_token_indices_maps[begin:end]):
        # Generate the feature_maps
        feature_maps = {}
        curr_sentence_rep.append(torch.mean(roberta_embedding[j, :sum(context_attention_mask[j]), :], dim=0))
        for (name, token_indices) in name_token_indices.items():
          token_features = roberta_embedding[j, token_indices, :]

          # Use max pooling to join the features
          agg_token_feature = torch.max(token_features, dim=0).values
          feature_maps[name] = agg_token_feature

        # Generate name pairs
        names = list(name_token_indices.keys())
        curr_sentence_name_pairs = [(m, n) for m in names for n in names if m != n]
        curr_datapoint_name_pairs += curr_sentence_name_pairs
        curr_datapoint_name_pairs_features += [torch.cat((feature_maps[x], feature_maps[y])) for (x, y) in curr_sentence_name_pairs]

      global_rep = torch.mean(torch.stack(curr_sentence_rep), dim=0)

      # Generate the pairs for this datapoint
      num_name_pairs = len(curr_datapoint_name_pairs)
      splits.append((0, num_name_pairs) if len(splits) == 0 else (splits[-1][1], splits[-1][1] + num_name_pairs))
      name_pairs += curr_datapoint_name_pairs
      name_pairs_features += curr_datapoint_name_pairs_features

    # Stack all the features into the same big tensor
    name_pairs_features = torch.cat((torch.stack(name_pairs_features), global_rep.repeat(len(name_pairs_features), 1)), dim=1)

    # Use MLP to extract relations between names
    name_pair_relations = self.relation_extraction(name_pairs_features)

    # Return the extracted relations and their corresponding symbols
    return (splits, name_pairs, name_pair_relations)

  def _extract_facts(self, splits, name_pairs, name_pair_relations, queries):
    context_facts, context_disjunctions, question_facts = [], [], []
    num_pairs_processed = 0

    # Generate facts for each context
    for (i, (begin, end)) in enumerate(splits):
      # First combine the name_pair features if there are multiple of them, using max pooling
      name_pair_to_relations_map = {}
      for (j, name_pair) in zip(range(begin, end), name_pairs[begin:end]):
        name_pair_to_relations_map.setdefault(name_pair, []).append(name_pair_relations[j])
      name_pair_to_relations_map = {k: torch.max(torch.stack(v), dim=0).values for (k, v) in name_pair_to_relations_map.items()}

      # Generate facts and disjunctions
      curr_context_facts = []
      curr_context_disjunctions = []
      for ((sub, obj), relations) in name_pair_to_relations_map.items():
        curr_context_facts += [(relations[k], (k, sub, obj)) for k in range(len(relation_id_map))]
        curr_context_disjunctions.append(list(range(len(curr_context_facts) - len(relation_id_map), len(curr_context_facts))))
      context_facts.append(curr_context_facts)
      context_disjunctions.append(curr_context_disjunctions)
      question_facts.append([queries[i]])

      # Increment the num_pairs processed for the next datapoint
      num_pairs_processed += len(name_pair_to_relations_map)

    # Return the facts generated
    return (context_facts, context_disjunctions, question_facts)

  def forward(self, x, phase='train'):
    (contexts, queries, context_splits) = x

    # Debug prints
    if self.debug:
      print(contexts)
      print(queries)

    # Go though the preprocessing, RoBERTa model forwarding, and facts extraction steps
    (clean_contexts, clean_context_splits, name_token_indices_maps) = self._preprocess_contexts(contexts, context_splits)
    (splits, name_pairs, name_pair_relations) = self._extract_relations(clean_contexts, clean_context_splits, name_token_indices_maps)
    (context_facts, context_disjunctions, question_facts) = self._extract_facts(splits, name_pairs, name_pair_relations, queries)

    transitivity_probs = torch.clamp(self.transitivity_probs, min=0, max=1)
    if phase == 'train':
      sampled_transitive_idx = torch.multinomial(transitivity_probs, self.sample_ct)
    else:
      _, sampled_transitive_idx = torch.topk(transitivity_probs, self.sample_ct)

    probs = transitivity_probs[sampled_transitive_idx]
    transitives = [all_possible_transitives[i] for i in sampled_transitive_idx]
    transitive_relations = [[(prob, relation) for prob, relation in zip(probs, transitives)] for _ in range(len(context_facts))]

    # Go though the preprocessing, RoBERTa model forwarding, and facts extraction steps
    # (splits, name_pairs, name_pair_relations) = self._extract_relations(clean_contexts, clean_context_splits, name_token_indices_maps)
    # (context_facts, context_disjunctions, question_facts) = self._extract_facts(splits, name_pairs, name_pair_relations, queries)

    # Run Scallop to reason the result relation
    result = self.reason(context=context_facts, transitive=transitive_relations, question=question_facts, disjunctions={"context": context_disjunctions})
    if self.scallop_softmax:
      result = torch.softmax(result, dim=1)

    # Return the final result
    return result


class Trainer:
  def __init__(self, train_loader, test_loader, device, model_dir, model_name, learning_rate, rule_learning_rate, constraint_weight, **args):
    self.device = device
    load_model = args.pop('load_model')
    self.decay_ct = args.pop('decay_ct')

    if load_model:
      new_model = CLUTRRModel(device=device, **args).to(device)
      loaded_model = torch.load(os.path.join(model_dir, model_name + '.latest.model'))
      new_model.tokenizer = loaded_model.tokenizer
      new_model.roberta_model = loaded_model.roberta_model
      new_model.transitivity_probs = loaded_model.transitivity_probs
      new_model.relation_extraction = loaded_model.relation_extraction
      self.model = new_model
    else:
      self.model = CLUTRRModel(device=device, **args).to(device)
    self.model_dir = model_dir
    self.model_name = model_name
    self.constraint_weight = constraint_weight
    # self.optimizer = optim.Adam(chain(self.model.parameters(), [self.model.transitivity_probs]) , lr=learning_rate)

    self.optimizer = optim.Adam(self.model.parameters() , lr=learning_rate)
    self.rule_optimizer = optim.Adam([self.model.transitivity_probs] , lr=rule_learning_rate)

    # self.optimizer = optim.RMSprop(self.model.parameters() , lr=learning_rate)
    # self.rule_optimizer = optim.RMSprop([self.model.transitivity_probs] , lr=rule_learning_rate)
    # self.lr_optim = ExponentialLR(self.optimizer, gamma=0.8)
    # self.lr_rule = ExponentialLR(self.rule_optimizer, gamma=0.8)

    self.train_loader = train_loader
    self.test_loader = test_loader
    self.min_test_loss = 10000000000.0
    self.max_accu = 0


  def loss(self, y_pred, y):
    result = y_pred['answer']
    ic = y_pred['violate_ic']
    (_, dim) = result.shape
    gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in y]).to('cuda')
    result_loss = nn.functional.binary_cross_entropy(result, gt)
    # ic_loss = 0.1 * nn.functional.nll_loss(ic[:, 1], torch.tensor([0] * ic.shape[0]).to('cuda'))
    ic_loss = args.constraint_weight * nn.functional.l1_loss(ic[:, 1], torch.tensor([0] * ic.shape[0]).to('cuda'))
    return result_loss + ic_loss

  def accuracy(self, y_pred, y):
    batch_size = len(y)
    result = y_pred['answer'].detach()
    pred = torch.argmax(result, dim=1)
    num_correct = len([() for i, j in zip(pred, y) if i == j])
    return (num_correct, batch_size)

  def train(self, num_epochs):
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
      self.rule_optimizer.zero_grad()
      y_pred = self.model(x, 'train')
      loss = self.loss(y_pred, y)
      total_loss += loss.item()
      loss.backward()
      self.optimizer.step()
      self.rule_optimizer.step()

      (num_correct, batch_size) = self.accuracy(y_pred, y)
      total_count += batch_size
      total_correct += num_correct
      correct_perc = 100. * total_correct / total_count
      avg_loss = total_loss / (i + 1)

      iterator.set_description(f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Accuracy: {total_correct}/{total_count} ({correct_perc:.2f}%)")
      # if i % self.decay_ct:
      #   self.model.update_decay()

  def test_epoch(self, epoch, save=True):
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

  def get_rules(self, sample_ct):
    pred_probs = self.model.transitivity_probs.reshape(-1)
    rules = sorted([(pred_prob, [id_relation_map[e] for e in r]) for pred_prob, r in zip(pred_probs, all_possible_transitives)], reverse=True)[:sample_ct]
    return rules

def pretty_print_rules(rules):
  rule_str = '\n'.join([f"{p}::{r}" for p, r in rules])
  print(rule_str)

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--dataset", type=str, default="data_089907f8")
  parser.add_argument("--model-name", type=str, default="clutrr_rule_and_facts_general")
  parser.add_argument("--load_model", type=bool, default=True)
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--seed", type=int, default=2345)
  parser.add_argument("--sample-ct", type=int, default=500)
  parser.add_argument("--learning-rate", type=float, default=0.00001)
  parser.add_argument("--rule-learning-rate", type=float, default=0.001)
  parser.add_argument("--prob-decay", type=float, default=1)
  parser.add_argument("--decay-ct", type=int, default=100)

  parser.add_argument("--constraint-weight", type=float, default=0.1)


  parser.add_argument("--num-mlp-layers", type=int, default=2)
  parser.add_argument("--provenance", type=str, default="difftopbottomkclauses")
  # parser.add_argument("--provenance", type=str, default="diffaddmultprob")
  parser.add_argument("--train-top-k", type=int, default=3)
  parser.add_argument("--test-top-k", type=int, default=3)
  parser.add_argument("--no-fine-tune-roberta", type=bool, default=False)
  parser.add_argument("--use-softmax", type=bool, default=True)
  parser.add_argument("--scallop-softmax", type=bool, default=False)
  parser.add_argument("--debug", action="store_true")
  parser.add_argument("--cuda", type=bool, default=True)
  parser.add_argument("--gpu", type=int, default=0)

  args = parser.parse_args()
  print(args)

  args.model_name = f"{args.model_name}_{args.sample_ct}"

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
  args.gpt_pred_path = os.path.join(data_root, 'gpt_rules.json')
  if not os.path.exists(model_dir): os.makedirs(model_dir)

  # Load the dataset
  (train_loader, test_loader) = clutrr_loader(data_root, args.dataset, args.batch_size)

  # Train
  trainer = Trainer(train_loader, test_loader, device, model_dir, args.model_name, args.learning_rate, constraint_weight = args.constraint_weight,
                    num_mlp_layers=args.num_mlp_layers, debug=args.debug, provenance=args.provenance, train_top_k=args.train_top_k, test_top_k=args.test_top_k,
                    use_softmax=args.use_softmax, no_fine_tune_roberta=args.no_fine_tune_roberta, scallop_softmax = args.scallop_softmax,load_model=args.load_model,
                    sample_ct=args.sample_ct, rule_learning_rate=args.rule_learning_rate, gpt_pred_path=args.gpt_pred_path, decay_ct = args.decay_ct,
                    prob_decay = args.prob_decay)
  # trainer.train(args.n_epochs)
  rules = trainer.get_rules(args.sample_ct)
  pretty_print_rules(rules)
  # trainer.test_epoch(0, False)
