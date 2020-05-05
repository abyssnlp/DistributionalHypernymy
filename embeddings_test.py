
# embeddings test
# pre-trained

import numpy as np
import pickle

#! Memory Error on Embeddings to Dict
def embeddings_to_dict(embeddings,dict_path):
    embeddings_dict={}
    with open(embeddings,'r') as f:
        for line in f:
            items=line.split()
            word=items[0]
            vector=np.asarray(items[1:])
            embeddings_dict[word]=vector
    with open(dict_path,'wb') as g:
        pickle.dump(embeddings_dict,g,protocol=pickle.HIGHEST_PROTOCOL)
                
embeddings_to_dict('/home/srawat/Documents/Distributional Hypernymy/enwiki_20180420_300d.txt','/home/srawat/Documents/Distributional Hypernymy/wiki2vec_dict.pkl')

# Alternate
from gensim import *

model=gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format('/home/srawat/Documents/Distributional Hypernymy/enwiki_20180420_300d.txt')

model.save('/home/srawat/Documents/Distributional Hypernymy/wiki2vec_300.model')

# load model
from gensim import *
model=models.keyedvectors.Word2VecKeyedVectors.load('/home/srawat/Documents/Distributional Hypernymy/wiki2vec_300.model')

# LogReg on BLESS
import pandas as pd
bless=pd.read_table('/home/srawat/Documents/Distributional Hypernymy/bless.tsv')

bless.columns=['word1','word2','label']
x_bless=bless[['word1','word2']]
y_bless=bless['label']
# X to vectors
x_bless['word1']=x_bless['word1'].map(lambda x:x.lower())
x_bless['word2']=x_bless['word2'].map(lambda x:x.lower())

# test vectors
w1_vectors=[]
w2_vectors=[]

for word in list(x_bless.word1):
    w1_vectors.append(model[word])
for word in list(x_bless.word2):
    try:
        w2_vectors.append(model[word])
    except KeyError:
        pass

# some words are OOV
# only put words that are both in the vocabulary
w1_invocab=[]
w2_invocab=[]
y_invocab=[]
for w1,w2,label in zip(bless.word1,bless.word2,bless.label):
    if w1 in model and w2 in model:
        w1_invocab.append(w1)
        w2_invocab.append(w2)
        y_invocab.append(label)
# filter out x_bless
x_bless=pd.DataFrame(w2_invocab,w1_invocab)
x_bless=x_bless.reset_index()
x_bless.columns=['word1','word2']
x_bless['word1']=x_bless['word1'].map(lambda x:model[x])
x_bless['word2']=x_bless['word2'].map(lambda x:model[x])

# create word vector by concatenating the sub word vectors

x_bless['concat']=x_bless['word1']+x_bless['word2']
x_bless=x_bless.drop(['concat'],axis=1)

test=pd.concat([x_bless['word1'],x_bless['word2']])
# y to binomial
y_test=[]
for item in y_invocab:
    if item==False:
        y_test.append(0)
    else:
        y_test.append(1)
y_invocab=y_test
y_invocab=pd.Series(y_invocab)
y_invocab=pd.Series(y_invocab,dtype=np.float32)

import itertools
#y_invocab=list(itertools.chain.from_iterable(y_invocab))

# Train, test split
from sklearn.model_selection import train_test_split
x_bless_train,x_bless_test,y_bless_train,y_bless_test=train_test_split(x_bless.concat,y_invocab,test_size=0.2)

x_bless_train.head()

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()

# classifier.fit(pd.concat([x_bless_train.word1,x_bless_train.word2],axis=1),y_bless_train) 

classifier.fit(list(x_bless_train),y_bless_train)

predictions=classifier.predict(list(x_bless_test))
from sklearn.metrics import accuracy_score,recall_score,precision_score,average_precision_score
accuracy_score(y_bless_test,predictions)
precision_score(y_bless_test,predictions)
recall_score(y_bless_test,predictions)
average_precision_score(y_bless_test,predictions)

# ## 2nd method concat the vectors in list
# list_concat=[]
# for i in range(len(x_bless)):
#     list_concat.append([x_bless.word1.iloc[i],x_bless.word2.iloc[i]])

# list_concat_train,list_concat_test,y_bless_train,y_bless_test=train_test_split(list_concat,y_invocab,test_size=0.2)
# classifier2=LogisticRegression()
# classifier2.fit(list_concat_train,y_bless_train)
# predictions_concat=classifier2.predict(list_concat_test)

# Concat vectors by stacking on top of each other
import numpy as np
stacked_vectors=[]
for i in range(len(x_bless)):
    stacked_vectors.append(np.vstack((x_bless.word1.iloc[i],x_bless.word2.iloc[i])))

# x_bless['concat_vectors']=np.concatenate((x_bless.word1,x_bless.word2),axis=None)

# concat vectors numpy
concat_vector=[]
for i in range(len(x_bless)):
    concat_vector.append(np.concatenate((x_bless.word1.iloc[i],x_bless.word2.iloc[i]),axis=None))
concat_bless_train,concat_bless_test,y_bless_train,y_bless_train=train_test_split(concat_vector,y_invocab,test_size=0.2)
classifier2=LogisticRegression()
classifier2.fit(list())