import sys
sys.path.append('../../alignment_tool')

import os
import utility
from langdetect import detect
import torch
import numpy as np
from scipy import stats
from scipy.special import expit, logit
from sentence_transformers import SentenceTransformer
source_units = []
query_units = []
root_path = (
  "/Users/tanzir5/Notebooks/pan_experiments/plagiarism_detection/"
  "pan-plagiarism-corpus-2011-1/external-detection-corpus/"
  )



def build_corpus(path, unit_len = 'paragraph', clip = int(1e9)):
  units = []
  count = 0
  for root_dirs, dirs, files in os.walk(path):
    for f in files:
      if len(units) > clip:
        break
      if f.endswith(".txt"):
        text = open(path + f).read()
        if detect(text) != 'en':
          print(detect(text))
        else:          
          if unit_len == 'paragraph':
            current_units = utility.get_paragraphs(text)
          elif unit_len == 'sentence':
            current_units = utility.get_sentences(text)
          else:
            assert(False)
          units.extend(current_units[5:-5])
  units = units[:clip]
  return units 
path = root_path + 'source-document/part1/'
source_units = build_corpus(path, unit_len = 'sentence', clip = 5)
path = root_path + 'suspicious-document/part2/'
query_units = build_corpus(path, unit_len = 'sentence', clip = 6)
print(len(source_units), len(query_units))


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
sbert = SentenceTransformer('msmarco-distilbert-cos-v5').to(DEVICE)
source_embs = sbert.encode(source_units)
query_embs = sbert.encode(query_units)
np.save("source_para_embs.npy",source_embs)
np.save("query_para_embs.npy",query_embs)