import argparse
import seaborn as sns
import sys
import utility 

import numpy as np
from preprocessor import Preprocessor
from aligner import Aligner
from utility import get_paragraphs, parse_text, print_matches, show_heat_map
import matplotlib.pyplot as plt

# Define the parser
def align_sequences(
                    seq1, seq2, 
                    unit1='paragraph', 
                    unit2='paragraph', 
                    sim='sbert', 
                    norm='sigmoid',
                    z_thresh=4,
                    clip=-1,
                    similarity_config={},
                    ignore=[set(), set()]
                    ): 
  similarity_config['sim'] = sim
  if norm == 'sigmoid':
    similarity_config['normalization'] = 'z_normalize_with_sigmoid'

  preprocessor = Preprocessor(seq1, seq2, 
                              token_size_of_A = unit1,
                              token_size_of_B = unit2,
                              threshold = float(z_thresh),
                              similarity_config = similarity_config,
                              clip_length = clip)

  aligner = Aligner(preprocessor.similarity_matrix, ignore)
  aligner.smith_waterman()
  return aligner
