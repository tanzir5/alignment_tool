# link for code: https://github.com/pan-webis-de/pan-code/blob/master/clef12/text-alignment/pan12-text-alignment-baseline.py
#!/usr/bin/env python
""" Plagiarism detection for near-duplicate plagiarism.
    This program provides the baseline for the PAN 2013 Plagiarism Detection
    task and can be used as outline for a plagiarism detection program.
"""
__author__ = 'Arnd Oberlaender'
__email__ = 'arnd.oberlaender at uni-weimar dot de'
__version__ = '1.1'

import utility
from preprocessor import Preprocessor
from aligner import Aligner

import os
import string
import sys
import xml.dom.minidom
import codecs
from tqdm import tqdm

# Const
# =====

DELETECHARS = ''.join([string.punctuation, string.whitespace])
LENGTH = 50

# Main
# ====
'''
def merge_matches(edge_seq):
    last_match_j = -1
    seq1_match = [-1, -1]
    seq2_match = [-1, -1]
    results = []
    for i in range(len(edge_seq)):
        if edge_seq[i] == -1:
            if seq2_match[1] != -1:
                results.append(seq1_match + seq2_match)
                seq2_match = [-1, -1]
        else:
            if seq2_match[1] + 1 == edge_seq[i]:
                seq2_match[1] += 1
                seq1_match[1] += 1
            elif seq2_match[1] == -1:
                seq1_match = [i, i]
                seq2_match = [edge_seq[i], edge_seq[i]]
            elif seq2_match[1] != -1 and edge_seq[i] != seq2_match[1]:
                results.append(seq1_match + seq2_match)
                seq2_match = [-1, -1]
            else:
                assert(False)
    if seq2_match[1] != -1:
        results.append(seq1_match + seq2_match)
    
    return results
'''

def align(susp, src, resultdir, sim='jaccard', thresh=0, z_thresh=4):
    if sim == 'jaccard':
        sim = 'overlapping_token_similarity_matrix'
    susp_fp = codecs.open(susp, 'r', 'utf-8')
    susp_text = susp_fp.read()
    susp_fp.close()

    src_fp = codecs.open(src, 'r', 'utf-8')
    src_text = src_fp.read()
    src_fp.close()
    
    text1 = utility.get_sentences(susp_text, add_start_position=True)
    text2 = utility.get_sentences(src_text, add_start_position=True)
    similarity_config = {}
    similarity_config['sim'] = sim
    similarity_config['normalization'] = 'z_normalize_with_sigmoid'

    text1_normal = [t[0] for t in text1]
    text2_normal = [t[0] for t in text2]
    print("similarity computation started")
    preprocessor = Preprocessor(text1_normal, text2_normal, 
                                token_size_of_A = 'sentence',
                                token_size_of_B = 'sentence',
                                threshold = float(z_thresh),
                                similarity_config = similarity_config,
                                )

    aligner = Aligner(preprocessor.similarity_matrix)
    aligner.smith_waterman(gap_start_penalty=-20000, gap_continue_penalty=-20000)
    matches = aligner.get_best_matches(thresh=thresh,top_k=-1)
    results = []
    for m in matches:
        #print(m)
        susp_bound_sent = (m[0][0], m[1][0])
        #print(susp_bound_sent)
        #print(len(text1))
        susp_bound_char = (
            text1[susp_bound_sent[0]][1],
            text1[susp_bound_sent[1]][1] + len(text1[susp_bound_sent[1]][0]) - 1
            )
        src_bound_sent = (m[0][1], m[1][1])
        src_bound_char = (
            text2[src_bound_sent[0]][1],
            text2[src_bound_sent[1]][1] + len(text2[src_bound_sent[1]][0]) - 1
            )
        #print(src_bound_sent)
        #print(src_bound_char, susp_bound_char)
        results.append((src_bound_char[0], src_bound_char[1], susp_bound_char[0], susp_bound_char[1]))
        #print(text1[susp_bound_sent[1]][0])
        #print(text2[src_bound_sent[1]][0])
    susp_name = os.path.split(susp)[1]
    src_name = os.path.split(src)[1]
    f = open(resultdir + susp_name.split('.')[0] + '-'
                      + src_name.split('.')[0] + '.txt', 'w')
    for result in results:
        f.write(' '.join([str(i) for i in result]))
        f.write('\n')
    f.close()

if __name__ == "__main__":
    """ Process the commandline arguments. We expect three arguments: The path
    pointing to the pairs file and the paths pointing to the directories where
    the actual source and suspicious documents are located.
    """
    print(len(sys.argv))
    print(sys.argv)
    if len(sys.argv) == 5:
        srcdir = sys.argv[2]
        suspdir = sys.argv[3]
        resultdir = sys.argv[4]
        if resultdir[-1] != "/":
            resultdir+="/"
        if os.path.exists(resultdir) == False:
            os.mkdir(resultdir)
        lines = open(sys.argv[1], 'r').readlines()
        for line in tqdm(lines):
            susp, src = line.split()
            print("Doing", line)
            align(os.path.join(suspdir, susp), os.path.join(srcdir, src), resultdir)
    else:
        print('\n'.join(["Unexpected number of commandline arguments.",
                         "Usage: ./pan12-plagiarism-text-alignment-example.py {pairs} {src-dir} {susp-dir} {result-dir}"]))