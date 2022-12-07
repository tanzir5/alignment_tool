import align_pipeline 
import numpy as np
from aligner import Aligner

def test_aligner():
  # testing alignment from similarity matrix
  r = 10
  c = 12
  sim_matrix = np.random.rand(r, c)
  #sim_matrix[2][11] = 200
  #sim_matrix[3][5] = 5
  #sim_matrix[4][6] = 4
  aligner = Aligner(sim_matrix)
  aligner.smith_waterman(no_gap=True)
  matches, e1, e2 = aligner.global_alignment()
  print(e1, e2)

def test_align_pipeline():
  #testing alignment from embedding and thresholds 
  mean = 0.093
  sigma = 0.088
  seq1 = np.random.rand(10, 768)
  seq2 = np.random.rand(12, 768)
  aligner = align_pipeline.align_sequences(
                seq1, seq2, 
                unit1='embedding', 
                unit2='embedding', 
                z_thresh=1,
                similarity_config={'mean':mean,'std':sigma},
                ignore=[set(), set()]
                ) 
  aligner.smith_waterman(no_gap=True)
  matches, e1, e2 = aligner.global_alignment()
  print(e1, e2)

#test_align_pipeline()
test_aligner()