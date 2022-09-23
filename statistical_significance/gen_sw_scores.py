import sys
sys.path.append('../../alignment_tool')

import glob
from tqdm import tqdm
import pandas as pd
from align_pipeline import align_sequences
import os
import xml.etree.ElementTree as ET


def update_sw_csv(book1, book2, score, path='sw_scores.csv'):
  df = pd.DataFrame({'book1':book1, 'book2':book2, 'score':score})
  df.to_csv('path', mode='a', index=False, header=False)

def get_seq(book_id):
  path = '/home/allekim/stonybook_data_guten/' + book_id + "/" + 'header_annotated.xml'
  
  tree = ET.parse(path)
  root = tree.getroot()
  body = root.find('body')
  seq = []
  for header in body.findall('header'):
    for para in header.findall('p'):
      seq.append(para.text)
  return seq


df = pd.read_csv('selected_books.csv')
books = df['book'].to_list()
print(type(books[0]))
exit(0)
lengths = []
for i in range(len(books)):
  lengths.append(len(get_seq(books[i])))

lengths = np.array(lengths)
print(np.min(lengths), np.median(lengths), np.max(lengths), np.mean(lengths), np.std(lengths))

#root_dir = './'
path = 'sw_scores.csv'
if not os.path.exists(path):
  df = pd.DataFrame({'book1':[], 'book2':[], 'score':[]})
  df.to_csv(path, index=False)


for i in range(len(books)):
  seq1 = get_seq(books[i])
  for j in range(i+1,len(books)):
    if ((df['book1'] == books[i]) & (df['book2'] == books[j])).any():
      continue
    seq2 = get_seq(books[j])
    dp = align_sequences(seq1, seq2)
    best = np.max(dp)
    update_sw_csv(best)