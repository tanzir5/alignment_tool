import spacy
from multiset import *
import numpy as np
from scipy import stats
from scipy.special import expit, logit
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from sklearn.feature_extraction.text import TfidfVectorizer
import copy
from tqdm import tqdm
import torch
import torchtext

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
nlp = spacy.load("en_core_web_sm")
all_stopwords = nlp.Defaults.stop_words
print("Running on ", DEVICE)

class Preprocessor:
  '''
  token_size can be character, word, sentence, paragraph, page, book_unit, chapter
  path_of_A should contain the path of the file containing text of file A. Each token of file A should be in a new line. 
  '''

  def __init__(
    self, tokens_of_A, tokens_of_B,
    token_size_of_A = 'paragraph',
    token_size_of_B = 'paragraph',
    threshold = 0,
    similarity_config = {},
    clip_length = -1):
    """Initializes preprocessor."""

    if clip_length != -1:
    	tokens_of_A = tokens_of_A[:clip_length]
    	tokens_of_B = tokens_of_B[:clip_length]
    self.sbert = None
    self.glove = None
    self.threshold = threshold
    self.valid_token_sizes = ["character", "word", "sentence", "paragraph", "page", 
                              "book_unit", "chapter", "embedding", "embedding_dir"]
    self.tokens_of_A = tokens_of_A
    self.tokens_of_B = tokens_of_B
    assert (token_size_of_A in self.valid_token_sizes and token_size_of_B in self.valid_token_sizes)
    self.similarity_matrix = self.get_similarity_matrix(tokens_of_A, tokens_of_B, token_size_of_A, token_size_of_B, threshold, similarity_config)

  def init_glove(self):
    glove_embedding_dim = 300
    self.glove = torchtext.vocab.GloVe(name="840B", 
                              dim= glove_embedding_dim) 

  def init_sbert(self):
    print("loading sbert")
    self.sbert = SentenceTransformer('msmarco-distilbert-cos-v5').to(DEVICE)
    print("loading done")

  def load_sbert_embedding(self, path):
    if os.path.exists(path):
      embs = np.load(path)
      return embs
    else:
      raise Exception(path + " does not exist.") 

  def get_similarity_matrix(self, tokens_of_A, tokens_of_B, token_size_of_A, token_size_of_B, threshold, similarity_config):
    if token_size_of_A == 'embedding_dir':
      if token_size_of_B == 'embedding_dir':
        tokens_of_A = self.load_sbert_embedding(tokens_of_A)
        tokens_of_B = self.load_sbert_embedding(tokens_of_B)
        token_size_of_A = 'embedding'
        token_size_of_B = 'embedding'
      else:
        return ['error']
    
    if token_size_of_A == 'character':
  	    if token_size_of_B == 'character':
  	        return self.get_similarity_matrix_char_char(tokens_of_A, tokens_of_B, similarity_config)
  	    else:
  	        return ['error']
    elif token_size_of_A == 'word':
  	    if token_size_of_B == 'word':
  	        return self.get_similarity_matrix_word_word(tokens_of_A, tokens_of_B, similarity_config)
  	    else:
  	        return ['error']
    elif token_size_of_A == 'embedding':
  	    if token_size_of_B == 'embedding':
  	        sim = util.cos_sim(tokens_of_A, tokens_of_B)
  	        #sim = util.dot_score(tokens_of_A, tokens_of_B)
  	        self.raw_sim_matrix = sim.detach().cpu().numpy()  
  	        return self.normalize(self.raw_sim_matrix, similarity_config)
  	    else:
  	        print("ERROR: A is embedding, B must be embedding instead of", tokens_size_of_B)
  	        return ['error']
    else:
  	    if token_size_of_B != 'character' and token_size_of_B != 'word': 
  	        return self.get_similarity_matrix_text_text(tokens_of_A, tokens_of_B, similarity_config)
  	    else:
  	        return ['error']
      

      
  def z_normalize_with_log(self, sim_matrix):
      #anything below +1 z score is -1
      #get z-score
      sim_matrix = stats.zscore(sim_matrix, axis = None)
      
      #take ln
      sim_matrix = np.clip(sim_matrix, a_min = 1e-9, a_max = None)
      sim_matrix = np.log2(sim_matrix)

      #clip eth below 1 to be 1
      sim_matrix = np.clip(sim_matrix, a_min = 1, a_max = None)
      
      #minmax scaling between -1 to 1
      saved_shape = sim_matrix.shape
      sim_matrix = minmax_scale(sim_matrix.flatten(), feature_range = (-1, 1))
      sim_matrix = np.reshape(sim_matrix, saved_shape)
      
      return sim_matrix

  def z_normalize_with_sigmoid(self, sim_matrix, threshold, sim_config = None):
      #anything below +2 z score is negative
      #get z-score
      if 'mean' in sim_config: 
          sim_matrix -= sim_config['mean']
          sim_matrix /= sim_config['std']
      else:
          sim_matrix = stats.zscore(sim_matrix, axis = None)
      #print("threshold", threshold)
      sim_matrix -= threshold
      sim_matrix = expit(sim_matrix) 
      sim_matrix *= 2
      sim_matrix -= 1
      assert((-1 <= sim_matrix).all() and (sim_matrix <= 1).all()) 
      
      return sim_matrix

  def z_normalize_with_tanh(self, sim_matrix):
      #anything below +1 z score is -1
      #get z-score
      sim_matrix = stats.zscore(sim_matrix, axis = None)
      sim_matrix -= 2
      sim_matrix = np.tanh(sim_matrix) 
      assert((-1 <= sim_matrix).all() and (sim_matrix <= 1).all()) 
      
      return sim_matrix

  def normalize(self, raw_sim_matrix, sim_config):
      if 'normalization' not in sim_config:
          return self.z_normalize_with_sigmoid(self.raw_sim_matrix, self.threshold, sim_config)
      elif sim_config['normalization'] == 'z_normalize_with_tanh': 
          return self.z_normalize_with_tanh(self.raw_sim_matrix, sim_config)
      elif sim_config['normalization'] == 'z_normalize_with_sigmoid' :
          return self.z_normalize_with_sigmoid(self.raw_sim_matrix, self.threshold, sim_config)
      elif sim_config['normalization'] == 'z_normalize_with_log' :
          return self.z_normalize_with_log(self.raw_sim_matrix, sim_config)
      else:
          print("NORMALIZE FUNCTION NOT SUPPORTED:", sim_config['normalization'])
          assert(False)

  def get_similarity_matrix_text_text(self, tokens_of_A, tokens_of_B, sim_config):
      if sim_config == None:
          self.raw_sim_matrix = self.overlapping_token_similarity_matrix(tokens_of_A, tokens_of_B)
          return self.raw_sim_matrix
      elif sim_config['sim'] in ['overlapping_token_similarity_matrix', 'jaccard', 'jaccard_sim']:
          self.raw_sim_matrix = self.overlapping_token_similarity_matrix(tokens_of_A, tokens_of_B)
      elif sim_config['sim'] == 'bert_embedding_sim':
          self.raw_sim_matrix = self.bert_embedding_sim(tokens_of_A, tokens_of_B)
      elif sim_config['sim'] == 'glove_embedding_sim':
          self.raw_sim_matrix = self.glove_embedding_sim(tokens_of_A, tokens_of_B)
      elif sim_config['sim'] == 'sbert':
          self.raw_sim_matrix = self.sbert_embedding_sim(tokens_of_A, tokens_of_B)
      elif sim_config['sim'] == 'tf_idf_sim':
          self.raw_sim_matrix = self.tf_idf_sim(tokens_of_A, tokens_of_B)
      elif sim_config['sim'] == 'overlapping_glove_sim' : 
          self.raw_sim_matrix = self.overlapping_glove_sim(tokens_of_A, tokens_of_B)
      elif sim_config['sim'] == 'hamming_sim' : 
          self.raw_sim_matrix = self.hamming_sim(tokens_of_A, tokens_of_B)
      
      else:
          print("SIM FUNCTION NOT SUPPORTED:", sim_config['sim'])
          assert(False)
      
      return self.normalize(self.raw_sim_matrix, sim_config)

  def tf_idf_sim(self, tokens_of_A, tokens_of_B):
      vectorizer = TfidfVectorizer(stop_words = {'english'})
      all_tokens = list(copy.deepcopy(tokens_of_A))
      all_tokens.extend(list(tokens_of_B))
      vectorizer = vectorizer.fit(all_tokens)
      
      tf_idf_A = vectorizer.transform(tokens_of_A).todense()
      tf_idf_B = vectorizer.transform(tokens_of_B).todense()
      
      #print(type(tf_idf_A))
      #print(tf_idf_A.shape)
      #print(tf_idf_B.shape)
      #print("Creating cosine similarities")
      ret = util.cos_sim(tf_idf_A, tf_idf_B)
      ret_np = ret.detach().cpu().numpy()  
      return ret_np

  def get_glove_embedding(self, tokens):
    ret = [[],[]]
    for token in tqdm(tokens):
      doc = nlp(token.lower())
      mean_emb = torch.zeros(300)
      max_emb = torch.zeros(300)
      min_emb = torch.ones(300) * 1e9
      for word in doc:
        if word.text not in all_stopwords and word.text.isalnum():
          cur = self.glove[word.text] 
          mean_emb += cur 
          max_emb = torch.max(max_emb, cur)
          min_emb = torch.min(min_emb, cur)
      min_max_emb = torch.zeros(300)
      for i in range(300):
        if max_emb[i] > -min_emb[i]:
          min_max_emb[i] = max_emb[i]
        else:
          min_max_emb[i] = min_emb[i]
      
      ret[0].append(mean_emb.unsqueeze(0))
      ret[1].append(min_max_emb.unsqueeze(0))
      
    ret[0] = torch.cat(ret[0])
    ret[1] = torch.cat(ret[1]) 
    return ret

  def glove_embedding_sim(self, tokens_of_A, tokens_of_B):
      self.init_glove()
      glove_A = self.get_glove_embedding(tokens_of_A)
      glove_B = self.get_glove_embedding(tokens_of_B)
      #print(type(glove_A[0]), type(glove_B[1]))
      #print(glove_A[0].shape, glove_A[1].shape)
      #print("Creating cosine similarities")
      ret_np = []
      for i in range(2):
        ret = util.cos_sim(glove_A[i], glove_B[i])
        ret_np.append(ret.detach().cpu().numpy())  
      
      return np.array(ret_np[0]) 

  def sbert_embedding_sim(self, tokens_of_A, tokens_of_B):
    if self.sbert is None:
    	self.init_sbert()
    print("sbert encoding starts")
    sbert_A = self.sbert.encode(tokens_of_A)
    sbert_B = self.sbert.encode(tokens_of_B)
    print("sbert encoding done")
    ret = util.cos_sim(sbert_A, sbert_B)
    ret_np = ret.detach().cpu().numpy()  
  	  
    return ret_np 


  def chunk(self, text, stride = 300, ch_len = 400):
      ret = []
      text = text.split(" ")
      for i in range(0,len(text), stride):
          ret.append(" ".join(text[i:min(len(text), i+ch_len)]))
      return ret

  def get_bert_embedding(self, tokens, batch_size = 16):
      torch.cuda.empty_cache()
      chunks_list = []
      boundary_chunks = []
      for (i,text) in enumerate(tokens):
          cur_chunks = self.chunk(text)
          boundary_chunks.append((len(chunks_list), len(chunks_list)+len(cur_chunks)))
          chunks_list.extend(cur_chunks)
          
      outputs = None
      encoded_input = bert_tokenizer(chunks_list, return_tensors = 'pt', padding = True, truncation = True, max_length = 512).to(device)
          
      #{'input_ids': [101, 2293, 3246, 3280, 102], 'token_type_ids': [0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1]}
      for i in range(0,len(chunks_list), batch_size):    
          
          #print("i =", i)
          #print(len(encoded_input['input_ids']), len(encoded_input['token_type_ids']), len(encoded_input['attention_mask']))

          tmp_out = bert_model(input_ids      = encoded_input['input_ids']     [i:min(len(chunks_list), i+batch_size)], 
                               token_type_ids = encoded_input['token_type_ids'][i:min(len(chunks_list), i+batch_size)],
                               attention_mask = encoded_input['attention_mask'][i:min(len(chunks_list), i+batch_size)])
          

          #emb = torch.mean(tmp_out.last_hidden_state, 1)
          #emb = torch.mean(tmp_out.last_hidden_state, 1).detach().clone()
          mx = torch.max(tmp_out.last_hidden_state, dim = 1)
          emb1 = mx.values.detach().clone()
          emb2 = tmp_out.last_hidden_state[:,0,:].detach().clone()
          assert(emb1.shape == emb2.shape)
          if i == 0:
              outputs = [emb1, emb2]
              #print("output shape", outputs[0].shape, emb1.shape)
          else:
              #print("output shape", outputs[0].shape, emb1.shape)
              outputs[0] = torch.cat((outputs[0],emb1))
              outputs[1] = torch.cat((outputs[1], emb2))

          #print("Memory used Before:",psutil.virtual_memory()[2])
          
          
          del tmp_out
          del emb1 
          del emb2 
          del mx             
          gc.collect() 
          #print("Memory used After:",psutil.virtual_memory()[2])
          
      #print("output shape", outputs[0].shape)
      #print("output shape", outputs[1].shape)
      embeddings = []
      embeddings.append(torch.zeros((len(boundary_chunks), outputs[0].shape[-1])))
      embeddings.append(torch.zeros((len(boundary_chunks), outputs[0].shape[-1])))
      #print("embeddings", embeddings[0].shape)
      for (i,boundary) in enumerate(boundary_chunks):
          embeddings[0][i] = torch.mean(outputs[0][boundary[0]:boundary[1]], dim = 0) 
          embeddings[1][i] = torch.mean(outputs[1][boundary[0]:boundary[1]], dim = 0) 

      embeddings[0] = embeddings[0].detach()  
      embeddings[1] = embeddings[1].detach()
      
      del outputs
      del encoded_input
      gc.collect()
      
    
      return embeddings 

  def get_bert_embedding_old(self, tokens):
      encoded_input = bert_tokenizer(tokens, return_tensors='pt', padding = True)
      output = bert_model(**encoded_input)
      ret = torch.mean(output.last_hidden_state, 1)
      
      return ret

  def bert_embedding_sim(self, tokens_of_A, tokens_of_B):
      print("Creating Bert Embeddings for A", len(tokens_of_A))
      bert_A = self.get_bert_embedding(tokens_of_A)
      #print("bert_A:",bert_A[0].shape)
      #print("bert_A:",bert_A[1].shape)
      print("Creating Bert Emebeddings for B", len(tokens_of_B))
      bert_B = self.get_bert_embedding(tokens_of_B)
      #print("bert_B:",bert_B[0].shape)
      #print("bert_B:",bert_B[1].shape)
      print("Creating cosine similarities")
      ret_np = []
      for i in range(2):
        ret = util.cos_sim(bert_A[i], bert_B[i])
        ret_np.append(ret.detach().cpu().numpy())  
        del ret

      del bert_A
      del bert_B
      gc.collect()
      torch.cuda.empty_cache()

      return ret_np 

  def get_words_multiset(self, token):
      words_multiset = Multiset()
      doc = nlp(token.lower())
      for word in doc:
          if word.text not in all_stopwords and word.text.isalnum():
              words_multiset.add(word.text)
      return words_multiset

  def get_jaccard_similarity(self, set_A, set_B):
      #return len((set_A & set_B)) / max(1,min(len(set_A),len(set_B))) 
      return len((set_A & set_B)) / max(1,len((set_A | set_B))) 

  def overlapping_token_similarity_matrix(self, tokens_of_A, tokens_of_B):
      sim_matrix = np.zeros((len(tokens_of_A), len(tokens_of_B)))
      print("Lengths of similarity matrix:" , len(tokens_of_A), len(tokens_of_B))
      words_multisets_A = []
      words_multisets_B = []
      for (i, token_A) in enumerate(tokens_of_A):
          words_multisets_A.append(self.get_words_multiset(token_A))
      
      for (i, token_B) in enumerate(tokens_of_B):
          words_multisets_B.append(self.get_words_multiset(token_B))
          
      for (i, token_A) in tqdm(enumerate(tokens_of_A)):
          #print(i)
          for (j, token_B) in enumerate(tokens_of_B):
              sim_matrix[i][j] = self.get_jaccard_similarity(words_multisets_A[i], words_multisets_B[j])
      #sim_matrix -= 0.2
      #sim_matrix *= 2
      return sim_matrix

  def _segment_into_seq(self, tokens, unit_size):
    if unit_size == 'word':
      words = []
      doc = nlp(tokens.lower())
      for word in doc:
        words.append(word.text)
      return words

  def _get_hamming_sim_seq(self, seq_A, seq_B):
    if len(seq_A) > len(seq_B):
      seq_A, seq_B = seq_B, seq_A
    window_len = len(seq_B) / len(seq_A)
    break_points = np.arange(0, len(seq_B), window_len)
    for i in range(len(break_points)):
      break_points[i] = int(round(break_points[i]))
    break_points = np.append(break_points, len(seq_B))
    break_points = break_points.astype('int')
    match_count = 0
    for i, unit_A in enumerate(seq_A):
      st = break_points[i]
      ed = break_points[i+1]
      if unit_A in set(seq_B[st:ed]):
        match_count += 1
    return match_count / len(seq_A)

  def _segment_into_seq_list(self, tokens, unit_size):
    seq_list = []
    for (i, token) in enumerate(tokens):
      seq_list.append(self._segment_into_seq(token, unit_size))
    return seq_list
  
  def hamming_sim(self, tokens_of_A, tokens_of_B, unit_size='word'):
    print("here")
    sim_matrix = np.zeros((len(tokens_of_A), len(tokens_of_B)))
    seq_A_list = self._segment_into_seq_list(tokens_of_A, unit_size)
    seq_B_list = self._segment_into_seq_list(tokens_of_B, unit_size)
    print("done")
    for (i, seq_A) in tqdm(enumerate(seq_A_list)):
      for (j, seq_B) in enumerate(seq_B_list):
        sim_matrix[i][j] = self._get_hamming_sim_seq(seq_A, seq_B)
    return sim_matrix

  def get_gloves_multiset(self, token):
      words_list = []
      doc = nlp(token.lower())
      for word in doc:
          if word.text not in all_stopwords and word.text.isalnum() and word.text in glove.stoi:
              idx = glove.stoi[word.text]
              if idx not in top_5:
                  top_5[idx] = nn_search.get_nns_by_item(idx, 5)
              words_list.extend(top_5[idx])
      return Multiset(words_list)

  def overlapping_glove_sim(self, tokens_of_A, tokens_of_B):
      sim_matrix = np.zeros((len(tokens_of_A), len(tokens_of_B)))
      #print("Lengths of similarity matrix:" , len(tokens_of_A), len(tokens_of_B))
      top_words_multisets_A = []
      top_words_multisets_B = []
      for (i, token_A) in enumerate(tokens_of_A):
          top_words_multisets_A.append(self.get_gloves_multiset(token_A))
      
      for (i, token_B) in enumerate(tokens_of_B):
          top_words_multisets_B.append(self.get_gloves_multiset(token_B))
      
          
      for (i, token_A) in enumerate(tokens_of_A):
          #print(i)
          for (j, token_B) in enumerate(tokens_of_B):
              sim_matrix[i][j] = self.get_jaccard_similarity(top_words_multisets_A[i], top_words_multisets_B[j])
      #sim_matrix -= 0.2
      #sim_matrix *= 2
      return sim_matrix

    