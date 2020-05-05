# Distributional Semantic Space creation script from Hypernym-LIBre
#! Multi-core Version
#! TODO: Add multi-core to vocab creation, dsm creation on split corpus
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
import multiprocessing as mp
from tqdm import tqdm

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

# Vocab to filter out extractions
def create_vocab_count(corpus):
    vocab_count={}
    tagger=nltk.PerceptronTagger()
    # only nouns,verbs and adjectives
    with open(corpus,'r') as f:
        for line in f:
            sentences=nltk.sent_tokenize(line)
            for sentence in sentences:
                sentence=clean_sentence(sentence).lower()
                words=sentence.split()
                pos_words=tagger.tag(words)
                words=[]
                for word,pos in pos_words:
                    if pos.startswith('NN') or pos.startswith('JJ') or pos.startswith('VB'):
                        words.append(word)
                words=[word for word in words if word not in stopwords]
                for word in words:
                    if word in vocab_count:
                        vocab_count[word]+=1
                    else:
                        vocab_count[word]=1
    return vocab_count

# Vocabulary for the corpus
def create_vocab(corpus):
    vocab={}
    #filter vocab
    vocab_count=create_vocab_count(corpus)
    vocab_words=dict(filter(lambda x:x[1]>100,vocab_count.items()))
    del vocab_count
    gc.collect()
    for key,value in vocab_words.items():
        if key in vocab:
            pass
        else:
            vocab[key]=len(vocab)
    return vocab

# Windowed Distributional space
def create_dsm(corpus,window):
    dok_dsm={}
    lemmatizer=nltk.stem.WordNetLemmatizer()
    vocab=create_vocab(corpus)
    t=[]
    c=[]
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
                t.extend(i)
                c.extend(j)
    gen_pairs=[]
    for a,b in zip(t,c):
        gen_pairs.append(tuple([a,b]))
    dok_dsm=dict(Counter(gen_pairs))
    i=[]
    j=[]
    for a,b in list(dok_dsm.keys()):
        i.append(a)
        j.append(b)
    v=list(dok_dsm.values())
    cooc_matrix=csr_matrix((v,(i,j)),shape=(len(vocab),len(vocab)),dtype=np.float64)
    save_npz('/home/srawat/Documents/Distributional Hypernymy/window_dsm_csr.npz')
    return cooc_matrix

# CLI execution
parser=argparse.ArgumentParser()

parser.add_argument('--corpus','-C',help='Corpus to create DSM')
parser.add_argument('--window','-W',help='Window size')

args=parser.parse_args()

if args.corpus and args.window:
    create_dsm(args.corpus,args.window)






