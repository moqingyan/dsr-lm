import json
import os

from tqdm import tqdm
from argparse import ArgumentParser

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

def get_gt_answer(gt_path):
  gt_rels = {}
  with open(gt_path, 'r') as f:
    for line in f:
      r1, r2, r3 = line[:-1].split(',')
      if not r1 in gt_rels:
        gt_rels[r1] = {}
      gt_rels[r1][r2] = r3
  return gt_rels

def get_filenames_and_answer(gt_rels):
  for rel1 in relation_id_map:
    for rel2 in relation_id_map:
      filename = f"{rel1}_{rel2}_result.json"
      if not rel1 in gt_rels:
        answer = None
      elif not rel2 in gt_rels[rel1]:
        answer = None
      else:
        answer = gt_rels[rel1][rel2]
      yield rel1, rel2, filename, answer

def check_correctness_of_file(data_dir, filename, answer):
  file_path = os.path.join(data_dir, filename)
  if not os.path.exists(file_path):
    pred = None
  else:
    response = json.load(open(file_path))
    for datapoint in response:
      datapoint = datapoint.replace(".", "").replace(",", "")
      preds = [l.strip() for l in datapoint.split() if l.strip() != ""]
      pred = preds[0] if len(preds) > 0 else ""

    # print(pred, answer)

  if answer is not None:
    if pred == answer:
      return pred, True
    if pred in relation_id_map:
      return pred, False
    return None, False
  else:
    return pred, None

def summarize_gpt(gt_path, gpt_dir):
  total_corrects = []
  gt_rels = get_gt_answer(gt_path)
  filename_answer_pairs = get_filenames_and_answer(gt_rels)
  gpt_preds = {}

  for rel1, rel2, filename, answer in filename_answer_pairs:
    pred, correct = check_correctness_of_file(gpt_dir, filename, answer)
    if correct is not None:
      total_corrects.append(correct)

    if not pred is None:
      if not rel1 in gpt_preds:
        gpt_preds[rel1] = {}
      gpt_preds[rel1][rel2] = pred

  correct_rate = sum([1 for i in total_corrects if i ]) / len(total_corrects)
  return gpt_preds, correct_rate

if __name__ == "__main__":

  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), '../../data'))
  gpt_dir = os.path.join(data_dir, 'gpt_composition_rules_prompt_Mary_D03')
  gt_path = os.path.join(data_dir, 'gt_rules.csv')
  gpt_path = os.path.join(data_dir, 'gpt_rules.json')
  gpt_preds, correct_rate = summarize_gpt(gt_path, gpt_dir)
  with open(gpt_path, 'w') as f:
    json.dump(gpt_preds, f)
  print(f'{gpt_dir}: {correct_rate}')
