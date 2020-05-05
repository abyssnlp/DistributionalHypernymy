# Distributional Semantic Space creation script from Hypernym-LIBre

# Windowed DSM indirectional

# Import packages
import nltk
import spacy
import itertools
from collections import Counter
import os
from scipy.sparse import csr_matrix,dok_matrix,save_npz
import argparse
import re
import gc
import numpy as np
import argparse
from tqdm import tqdm
import pickle
tagger=nltk.PerceptronTagger()
lemmatizer=nltk.stem.WordNetLemmatizer()
stopwords=nltk.corpus.stopwords.words('english')

# Pre-procesing
def clean_sentence(sentence):
    new_sent=[]
    words=sentence.split()
    words=list(itertools.chain.from_iterable([w.split(',') for w in words]))
    words=list(itertools.chain.from_iterable([w.split('-') for w in words]))
    for word in words:
        new_sent.append(''.join(w for w in word if w.isalpha()))
    return re.sub('\s\s+',r' ',' '.join(new_sent).strip())

# Windowed Distributional space
def create_dsm(corpus,window):
    dok_dsm={}
    lemmatizer=nltk.stem.WordNetLemmatizer()
    with open('/home/srawat/Documents/UMBC+Wiki/umbc_wiki_vocab.pkl','rb') as f:
        vocab=pickle.load(f)

    with open(corpus,'r') as f:
        for line in tqdm(f):
            sentences=nltk.sent_tokenize(line)
            for sentence in sentences:
                sentence=clean_sentence(sentence).lower()
                words=sentence.split()
                words=[lemmatizer.lemmatize(word) for word in words]
                words=[word for word in words if word in vocab]
                
                words=[vocab.get(word) for word in words]
                i=[]
                j=[]
                for index in range(len(words)):
                    if index<=window:
                        context=[]
                        context.append(words[:index])
                        context.append(words[index+1:index+window+1])
                        context=list(itertools.chain.from_iterable(context))
                        context=[x for x in context if x!=[]]
                        target=words[index]
                        for con in context:
                            i.append(target)
                            j.append(con)
                    elif index>window and index<=len(words)-window-1:
                        context=[]
                        context.append(words[index-2:index])
                        context.append(words[index+1:index+window+1])
                        context=list(itertools.chain.from_iterable(context))
                        context=[x for x in context if x!=[]]
                        target=words[index]
                        for con in context:
                            i.append(target)
                            j.append(con)
                    elif index>len(words)-window-1:
                        context=[]
                        context.append(words[index-2:index])
                        context.append(words[index+1:])
                        context=list(itertools.chain.from_iterable(context))
                        context=[x for x in context if x!=[]]
                        target=words[index]
                        for con in context:
                            i.append(target)
                            j.append(con)
                with open('/home/srawat/Documents/UMBC+Wiki/dsm_files/target.txt','a') as f:
                    for item in i:
                        f.write(str(item))
                        f.write('\n')
                with open('/home/srawat/Documents/UMBC+Wiki/dsm_files/context.txt','a') as g:
                    for item2 in j:
                        g.write(str(item2))
                        g.write('\n')
                
    # gen_pairs=[]
    # for a,b in zip(t,c):
    #     gen_pairs.append(tuple([a,b]))
    # dok_dsm=dict(Counter(gen_pairs))
    # i=[]
    # j=[]
    # for a,b in list(dok_dsm.keys()):
    #     i.append(a)
    #     j.append(b)
    # v=list(dok_dsm.values())
    # cooc_matrix=csr_matrix((v,(i,j)),shape=(len(vocab),len(vocab)),dtype=np.float64)
    # save_npz('/home/srawat/Documents/Distributional Hypernymy/window_dsm_csr.npz',cooc_matrix)
    # # save_npz('/home/srawat/Documents/UMBC+Wiki/test/test_dsm.npz',cooc_matrix)
    

# CLI execution
parser=argparse.ArgumentParser()

parser.add_argument('--corpus','-C',help='Corpus to create DSM')
parser.add_argument('--window','-W',help='Window size')

args=parser.parse_args()

if args.corpus and args.window:
    create_dsm(args.corpus,int(args.window))

# from scipy.sparse import load_npz
# cooc_test=load_npz('/home/srawat/Documents/UMBC+Wiki/test/test_dsm.npz')


# array test
test=[1,2,3,4,5]
with open('/home/srawat/Documents/UMBC+Wiki/test/test_target.txt','w') as f:
    for item in test:
        f.write(str(item))
        f.write('\n')