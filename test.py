
import re
import string
import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import gc

punct=string.punctuation
stopwords=stopwords.words('english')

with open('test_corpus.txt','r') as f:
    data=f.read()

# corpus cleaning
def clean_corpus(data):
    new_sentences=[]
    sentences=nltk.sent_tokenize(data)
    for sentence in sentences:
        sentence=re.sub(r'[^a-zA-z\s]',r'',sentence)
        sentence=re.sub("(?<=[a-z])'(?=[a-z])", "", sentence)
        sentence=sentence.lower()
        words=sentence.split()
        words=[word for word in words if word not in punct]
        words=[word for word in words if word not in stopwords]
        new_sentences.append(words)
    del sentences
    gc.collect()
    return new_sentences

new_sentences=clean_corpus(data)


model.load('test_model')




