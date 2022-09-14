import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
def k_largest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[:-k-1:-1]
    return np.column_stack(np.unravel_index(idx, a.shape))

#waterman_matrix = np.random.rand(4000, 3000)

seq_A = np.random.rand(13)
seq_B = np.random.rand(200)
window_len = len(seq_B) / len(seq_A)
break_points = np.arange(0, len(seq_B), window_len)
for i in range(len(break_points)):
  break_points[i] = int(round(break_points[i]))
print(window_len)
print(len(break_points))
print(break_points)