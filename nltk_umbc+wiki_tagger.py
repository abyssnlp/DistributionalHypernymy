# import spacy
import multiprocessing as mp
import nltk
import re
from tqdm import tqdm


# nlp=spacy.load('en_core_web_sm',disable=['ner'])
# nlp.max_length=1000000000

# tagger=nltk.tag.PerceptronTagger()
# test='Countries like Spain and France'
# tagged=tagger.tag(test.split())

# for word,pos in tagged:
#     print(word)

def process_file(filename):
    tagger=nltk.tag.PerceptronTagger()
    with open(filename,'r') as f:
        interim=[]
        data=f.read()
        sentences=nltk.sent_tokenize(data)
        for sentence in sentences:
            tagged_sent=[]
            words=sentence.split()
            words=tagger.tag(words)
            for word,pos in words:
                tagged_sent.append(word+'-'+pos)
            interim.append(' '.join(tagged_sent))
        interim='\n'.join(interim)
    with open('/home/srawat/Documents/UMBC+Wiki/corpus_tagged.txt','a') as g:
        g.write(interim)

import os
dir='/home/srawat/Documents/UMBC+Wiki/data'
files=[dir+'/'+f for f in os.listdir(dir) if f.startswith('data')]

pool=mp.Pool(processes=10)
extractions=pool.imap(process_file,files)

# for filee in tqdm(files):
#     process_file(filee)



