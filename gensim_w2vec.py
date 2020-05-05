# gensim based word2vec from UMBC+Wiki
import os
data_dir='/home/srawat/Documents/UMBC+Wiki/combined_corpus.txt'

import gensim
import nltk
import itertools
import re

def clean_sentence(sentence):
        new_sent=[]
        words=sentence.split()
        words=list(itertools.chain.from_iterable([w.split(',') for w in words]))
        words=list(itertools.chain.from_iterable([w.split('-') for w in words]))
        for word in words:
            new_sent.append(''.join(w for w in word if w.isalpha()))
        return re.sub('\s\s+',r' ',' '.join(new_sent).strip())
 
class CorpusReader(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        tagger=nltk.tag.PerceptronTagger()
        lemmatizer=nltk.stem.WordNetLemmatizer()
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                sentences=nltk.sent_tokenize(line)
                for sentence in sentences:
                    sentence=clean_sentence(sentence).lower()
                    words=sentence.split()
                    pos_words=tagger.tag(words)
                    words=[]
                    for word,pos in pos_words:
                        if pos.startswith('NN') or pos.startswith('JJ') or pos.startswith('VB'):
                            words.append(word)
                    words=[lemmatizer.lemmatize(word) for word in words]
                    yield words            
 
sentences = CorpusReader('/home/srawat/Documents/UMBC+Wiki/data/') 
model = gensim.models.Word2Vec(sentences,size=300,min_count=2,workers=12,iter=10,window=5)
model.save('/home/srawat/Documents/Distributional Hypernymy/gensim_umbc_w2vec_window5')













