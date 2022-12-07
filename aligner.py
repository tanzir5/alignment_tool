import numpy as np

INF = 1e9
def collides(a_st, a_ed, b_st, b_ed):
  return (a_st <= b_st <= a_ed) or (a_st <= b_ed <= a_ed)
    
class Aligner:
  def __init__(self, similarity_matrix, ignore = [set(), set()]):
      self.similarity_matrix = similarity_matrix
      self.ignore = ignore


  def global_prior(self, b, m):
      mu_b = math.ceil(b * self.N_M/self.N_B)
      exponent = (m - mu_b) * (m - mu_b) / (2*self.N_M*self.N_M)
      exponent *= -1
      return math.exp(exponent)*0.0125

  def smith_waterman(self, no_gap = False, gap_start_penalty = -0.9, gap_continue_penalty = -0.5, use_global_prior = False):
      if no_gap:
        gap_start_penalty = -INF
        gap_continue_penalty = -INF
      self.dp = np.zeros((self.similarity_matrix.shape[0]+1, self.similarity_matrix.shape[1]+1, 2, 2))
      if use_global_prior:
          self.N_M = self.dp.shape[0]
          self.N_B = self.dp.shape[1]
      self.parent = {}
      self.gap_start_penalty = gap_start_penalty
      self.gap_continue_penalty = gap_continue_penalty
      n = self.similarity_matrix.shape[0]
      m = self.similarity_matrix.shape[1]
      
      #Align n text units of first book against m text units of second book 
      #gc => gap_continue
      #similarity matrix uses 0 based indexing, dp(alone) uses 1 based indexing
      #print(n,m)
      for i in range(n+1):
        for j in range(m+1):
          for gc_1 in range(2):
            for gc_2 in range(2):
              #base case
              if i == 0 or j == 0:
                  self.dp[i][j][gc_1][gc_2] = 0
                  self.parent[(i, j, gc_1, gc_2)] = (-1, -1, -1, -1)
                  continue
              
              #Recurrence
              best = 0
              self.parent[(i, j, gc_1, gc_2)] = (-1, -1, -1, -1)
              if i in self.ignore[0]:
                  parent_value = self.dp[i-1][j][gc_1][gc_2]
                  self.dp[i][j][gc_1][gc_2] = parent_value
                  self.parent[(i, j, gc_1, gc_2)] = (i-1, j, gc_1, gc_2)
              elif j in self.ignore[1]:
                  parent_value = self.dp[i][j-1][gc_1][gc_2]
                  self.dp[i][j][gc_1][gc_2] = parent_value
                  self.parent[(i, j, gc_1, gc_2)] = (i, j-1, gc_1, gc_2)
              else:
                if gc_1 == 1:
                    #Try gaps
                    #continue gap for first text
                    #best = max(best, self.dp[i-1][j][gc_1][gc_2] + gap_continue_penalty)
                    parent_value = self.dp[i-1][j][gc_1][gc_2] + gap_continue_penalty
                    if best < parent_value:
                        best = parent_value
                        self.parent[(i, j, gc_1, gc_2)] = (i-1, j, gc_1, gc_2)
                else:
                    #start new gap for first text
                    #best = max(best, self.dp[i-1][j][1][gc_2] + gap_start_penalty)
                    parent_value = self.dp[i-1][j][1][gc_2] + gap_start_penalty
                    if best < parent_value:
                        best = parent_value
                        self.parent[(i, j, gc_1, gc_2)] = (i-1, j, 1, gc_2)
                if gc_2 == 1:
                    #continue gap for second text
                    #best = max(best, self.dp[i][j-1][gc_1][gc_2] + gap_continue_penalty)
                    parent_value = self.dp[i][j-1][gc_1][gc_2] + gap_continue_penalty
                    if best < parent_value:
                        best = parent_value
                        self.parent[(i, j, gc_1, gc_2)] = (i, j-1, gc_1, gc_2)
                else:
                    #start new gap for second text
                    #best = max(best, self.dp[i][j-1][gc_1][1] + gap_start_penalty)
                    parent_value = self.dp[i][j-1][gc_1][1] + gap_continue_penalty
                    if best < parent_value:
                        best = parent_value
                        self.parent[(i, j, gc_1, gc_2)] = (i, j-1, gc_1, gc_2)
                #Try matching
                #best = max(best, self.dp[i-1][j-1][0][0] + self.similarity_matrix[i-1][j-1])
                if use_global_prior:
                    # 1 <= gb <= 2 always true
                    gb = self.global_prior(j, i) # if positive sim, higher is better
                else:
                    gb = 0
                parent_value = self.dp[i-1][j-1][0][0] + self.similarity_matrix[i-1][j-1] + gb 
                
                if best < parent_value:
                    best = parent_value
                    self.parent[(i, j, gc_1, gc_2)] = (i-1, j-1, 0, 0)
              
                self.dp[i][j][gc_1][gc_2] = best
                    

  def set_global_align_variables(self):
      n = self.dp.shape[0]
      m = self.dp.shape[1]
      if n > m:
          #print("ERRROR!\nThe first one should be the smaller sequence.")
          pass

      self.dp_2d = np.zeros((n,m))
      self.edge_seq_1 = np.zeros(n)
      self.edge_seq_1.fill(-1)
      self.edge_seq_2 = np.zeros(m)
      self.edge_seq_2.fill(-1)
      
      all_cells = []
      for i in range(n):
          for j in range(m):
              self.dp_2d[i][j] = self.dp[i][j][0][0]
              all_cells.append((self.dp_2d[i][j], i, j))
      all_cells.sort(reverse = True)
      return all_cells

  def traverse(self, i, j, thresh, one_to_many):
      '''Finds the best match starting from (i, j) backwards.
      Args:
        i = index of 1st sequence from which backward search will start.
        j = index of 2nd sequence from which backward search will start.
        thresh = minimum threshold similarity required for matching two components of two sequences.
        one_to_many = if one component can be matched to multiple components. 
      Returns:
        Tuple(st,ed) where,
          st[0] -> start index of the match for the first sequence.
          st[1] -> start index of the match for the second sequence.
          ed[0] -> end index of the match for the first sequence.
          ed[1] -> end index of the match for the second sequence.
      
      All indices are inclusive.
      Dp uses 1 based indexing, but the returned indices are 0-based.
      
      thresh and one_to_many are useful for global alignment and creating self.edge_seq_1
      and self.edge_seq_2.
      '''
      gc_1 = 0
      gc_2 = 0
      ed = (i-1, j-1)
      overlaps = False
      while i > 0 and j > 0:
          if self.dp[i][j][gc_1][gc_2] == 0:
              break
          if self.edge_seq_2[j] != -1 or self.edge_seq_1[i] != -1 :
            overlaps = True
            break
          par = self.parent[(i, j, gc_1, gc_2)]
          if par[0] == i-1 and par[1] == j-1 and self.similarity_matrix[i-1][j-1] > thresh:
              #print(self.similarity_matrix[i-1][j-1])
              if one_to_many == False and self.edge_seq_2[j] == -1 and self.edge_seq_1[i] == -1 :
                  self.edge_seq_2[j] = i
                  self.edge_seq_1[i] = j
              elif (one_to_many == True):
                  if self.edge_seq_2[j] == -1:
                      self.edge_seq_2[j] = i
                  if self.edge_seq_1[i] == -1:
                      self.edge_seq_1[i] = j
                  
          '''
          if (self.edge_seq_2[j] == -1 
          and (one_to_many == True or self.edge_seq_1[i] == -1) 
          and par[0] == i-1 
          and par[1] == j-1):
              self.edge_seq_2[j] = i
              self.edge_seq_1[i] = j
          '''
          
          i, j, gc_1, gc_2 = par
      st = (i, j)
      return (st, ed, overlaps)

  def global_alignment(self, thresh = -0.2, one_to_many = False, top_k = -1):
    #print("here")
    all_cells = self.set_global_align_variables()
    #print(len(all_cells), "len")
    matches = []
    for count, cell in enumerate(all_cells):
        #print(cell)
        if len(matches) == top_k or cell[0] == 0:
            break
        i, j = cell[1], cell[2]
        if ((one_to_many == False and self.edge_seq_2[j] == -1 and self.edge_seq_1[i] == -1) or
            (one_to_many == True and (self.edge_seq_2[j] == -1 or self.edge_seq_1[i] == -1))
            ):
          candidate_match = self.traverse(i, j, thresh, one_to_many)
          collision = candidate_match[2]
          if not collision:
            for m in matches:
              if ( collides(m[0][0][0], m[0][1][0], candidate_match[0][0], candidate_match[1][0]) or
                   collides(m[0][0][1], m[0][1][1], candidate_match[0][1], candidate_match[1][1])
                 ):
                collision = True
                break
          if not collision:
            matches.append((candidate_match, cell[0]))
          #print("cell", cell)
        #if self.edge_seq_2[j] == -1 and (one_to_many == True or self.edge_seq_1[i] == -1):
        #    self.traverse(i,j,one_to_many)
    
    for i in range(1, len(self.edge_seq_1)):
        if self.edge_seq_1[i] >= 0:
          self.edge_seq_1[i-1] = self.edge_seq_1[i] - 1
        else:
          self.edge_seq_1[i-1] = self.edge_seq_1[i]
    for i in range(1, len(self.edge_seq_2)):
        if self.edge_seq_2[i] >= 0:
            self.edge_seq_2[i-1] = self.edge_seq_2[i] - 1
        else:
            self.edge_seq_2[i-1] = self.edge_seq_2[i]
    
    self.edge_seq_1 = np.delete(self.edge_seq_1, -1)
    self.edge_seq_2 = np.delete(self.edge_seq_2, -1)

    return matches, self.edge_seq_1, self.edge_seq_2

  def get_best_matches(self, thresh=-0.2, top_k=2):
    return self.global_alignment(thresh=thresh, top_k = top_k)
