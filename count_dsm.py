# convert corpus to dsm

with open('/home/srawat/Documents/UMBC+Wiki/test/test_corpus.txt','r') as f:
    data= f.read()

import nltk
sentences=nltk.sent_tokenize(data)
import re

test=r'Countries``` like... fantastic device...\\\\ ?????'
re.sub(r'[^a-zA-z\s]+',r'',test)

len(test.split())-1
test.split()[4]

# alpha=lambda w:w.isalpha()
new_test=[]
test.split()
for word in test.split():
    new_test.append(''.join(w for w in word if w.isalpha()))
' '.join(new_test).strip()

def alpha(sentence):
    new_sent=[]
    for word in sentence.split():
        new_sent.append(''.join(w for w in word if w.isalpha()))
    return ' '.join(new_sent).strip()

import itertools
def alt_alpha(sentence):
    new_sent=[]
    words=sentence.split()
    words=list(itertools.chain.from_iterable([w.split(',') for w in words]))
    words=list(itertools.chain.from_iterable([w.split('-') for w in words]))
    for word in words:
        new_sent.append(''.join(w for w in word if w.isalpha()))
    return ' '.join(new_sent).strip()

# alt with whitespace sub
def alt_alt_alpha(sentence):
    new_sent=[]
    words=sentence.split()
    words=list(itertools.chain.from_iterable([w.split(',') for w in words]))
    words=list(itertools.chain.from_iterable([w.split('-') for w in words]))
    for word in words:
        new_sent.append(''.join(w for w in word if w.isalpha()))
    return re.sub('\s\s+',r' ',' '.join(new_sent).strip())

# test alpha
test='Exercise-oriented methods are for mind,body,spirit.'
alt_alpha(test)
alpha(test)

test2='computer-to-computer,'
test2.split(',')[0].split('-')
alt_alpha(test2)

# all words vocab
def create_vocab(corpus):
    vocab={}
    # all words vocab
    with open(corpus,'r') as f:
        for line in f:
            sentences=nltk.sent_tokenize(line)
            for sentence in sentences:
                sentence=alt_alpha(sentence)
                words=sentence.split()
                for word in words:
                    if word in vocab:
                        pass
                    else:
                        vocab[word]=len(vocab)
    return vocab

# test
tagger=nltk.PerceptronTagger()
tagger.tag(['an','amazing','thing','was','found'])[0][0]

# POS tagged vocab with only NN, VB and JJ
# TODO: Remove Stop Words
def create_vocab_pos(corpus):
    vocab={}
    tagger=nltk.PerceptronTagger()
    # only nouns,verbs and adjectives
    with open(corpus,'r') as f:
        for line in f:
            sentences=nltk.sent_tokenize(line)
            for sentence in sentences:
                sentence=alt_alpha(sentence).lower()
                words=sentence.split()
                pos_words=tagger.tag(words)
                words=[]
                for word,pos in pos_words:
                    if pos.startswith('NN') or pos.startswith('JJ') or pos.startswith('VB'):
                        words.append(word)
                for word in words:
                    if word in vocab:
                        pass
                    else:
                        vocab[word]=len(vocab)
    return vocab

# POS tagged vocab without stopwords and with count
# TODO: Lemmatize the vocab words
stopwords=nltk.corpus.stopwords.words('english')
def create_vocab_count(corpus):
    vocab={}
    tagger=nltk.PerceptronTagger()
    # only nouns,verbs and adjectives
    with open(corpus,'r') as f:
        for line in f:
            sentences=nltk.sent_tokenize(line)
            for sentence in sentences:
                sentence=alt_alpha(sentence).lower()
                words=sentence.split()
                pos_words=tagger.tag(words)
                words=[]
                for word,pos in pos_words:
                    if pos.startswith('NN') or pos.startswith('JJ') or pos.startswith('VB'):
                        words.append(word)
                words=[word for word in words if word not in stopwords]
                for word in words:
                    if word in vocab:
                        vocab[word]+=1
                    else:
                        vocab[word]=1
    return vocab

# POS tagged vocab without stopwords, lemmatized words and directional, no count
# Sparse matrix requires numerical row and col value( i and j), change every word
# in the vocabulary to having both 'l' as well as 'r' suffixes
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
def create_vocab_directional(corpus):
    vocab={}
    tagger=nltk.PerceptronTagger()
    # only nouns,verbs and adjectives
    with open(corpus,'r') as f:
        for line in f:
            sentences=nltk.sent_tokenize(line)
            for sentence in sentences:
                sentence=alt_alpha(sentence).lower()
                words=sentence.split()
                pos_words=tagger.tag(words)
                words=[]
                for word,pos in pos_words:
                    if pos.startswith('NN') or pos.startswith('JJ') or pos.startswith('VB'):
                        words.append(word)
                words=[lemmatizer.lemmatize(word) for word in words if word not in stopwords]
                for word in words:
                    if word in vocab:
                        pass
                    else:
                        vocab[word]=len(vocab)
                        vocab[str(word)+'/l']=len(vocab)
                        vocab[str(word)+'/r']=len(vocab)
    return vocab
# TODO: Deal with filtering problem with sparse directionality

vocab=create_vocab('/home/srawat/Documents/UMBC+Wiki/test/test_corpus.txt')

vocab=create_vocab_pos('/home/srawat/Documents/UMBC+Wiki/test/test_corpus.txt')

vocab=create_vocab_count('/home/srawat/Documents/UMBC+Wiki/test/test_corpus.txt')

vocab_directional=create_vocab_directional('/home/srawat/Documents/UMBC+Wiki/test/test_corpus.txt')
# Filter vocab
# Only words with freq>100
new_vocab=dict(filter(lambda x:x[1]>100,vocab.items()))

# n=0
# for word,count in vocab.items():
#     if count>100:
#         n+=1

# test
test='Countries like Spain, France and Germany have been talking about trade arrangements.'
words=alt_alpha(test).split()
# words=[vocab.get(word) for word in words]
i=[]
j=[]
v=[]
for index in range(len(words)):
    if index<=2:
        context=[]
        context.append(words[:index])
        context.append(words[index+1:index+3])
        context=list(itertools.chain.from_iterable(context))
        context=[x for x in context if x!=[]]
        target=words[index]
        for con in context:
            i.append(target)
            j.append(con)
    elif index>2 and index<=len(words)-3:
        context=[]
        context.append(words[index-2:index])
        context.append(words[index+1:index+3])
        context=list(itertools.chain.from_iterable(context))
        context=[x for x in context if x!=[]]
        target=words[index]
        for con in context:
            i.append(target)
            j.append(con)
    elif index>len(words)-3:
        context=[]
        context.append(words[index-2:index])
        context.append(words[index:])
        context=list(itertools.chain.from_iterable(context))
        context=[x for x in context if x!=[]]
        target=words[index]
        for con in context:
            i.append(target)
            j.append(con)

# for a,b in zip(i,j):
#     print(a,b)
from collections import Counter

# test
i=[1,6,8,4,2,6,8]
j=[6,9,3,2,7,9,3]
pairs=[]
for a,b in zip(i,j):
    pairs.append(tuple([a,b]))
test_dict=dict(Counter(pairs))
i=[]
j=[]
for a,b in list(test_dict.keys()):
    i.append(a)
    j.append(b)

v=list(test_dict.values())
csr_matrix((v,(i,j)),shape=(len(vocab),len(vocab)),dtype=np.float64)

# Windowed DSM 
from nltk.stem import WordNetLemmatizer
from scipy.sparse import *
import numpy as np
def create_dsm(corpus,vocab,window):
    dok_dsm={}
    lemmatizer=WordNetLemmatizer()
    t=[]
    c=[]
    with open(corpus,'r') as f:
        for line in f:
            sentences=nltk.sent_tokenize(line)
            for sentence in sentences:
                sentence=alt_alpha(sentence)
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
    return cooc_matrix

cooc_matrix=create_dsm('/home/srawat/Documents/UMBC+Wiki/test/test_corpus.txt',new_vocab,window=5)   
cooc_matrix=cooc_matrix.todok()


# test directionality DSM
i=[]
j=[]
window=2
test='Countries like France and Spain. The trade agreements were made.'
words=test.split()
words=[word for word in words if word in vocab_directional]
for index in range(len(words)):
    if index<=window:
        left_context=[]
        right_context=[]
        for word in words[:index]:
            left_context.append(str(word)+'/l')
        for word in words[index+1:index+window+1]:
            right_context.append(str(word)+'/r')
        context=[]
        context.append(left_context)
        context.append(right_context)
        context=[x for x in context if x!=[]]

        context=list(itertools.chain.from_iterable(context))
        
        target=words[index]
        for con in context:
            i.append(vocab_directional.get(target))
            j.append(vocab_directional.get(con))
    

# Directional Context DSM
def create_dsm_directional(corpus,vocab,window):
    dok_dsm={}
    lemmatizer=WordNetLemmatizer()
    t=[]
    c=[]
    with open(corpus,'r') as f:
        for line in f:
            sentences=nltk.sent_tokenize(line)
            for sentence in sentences:
                sentence=alt_alpha(sentence)
                words=sentence.split()
                words=[lemmatizer.lemmatize(word) for word in words]
                words=[word for word in words if word in vocab]
                
                #words=[vocab.get(word) for word in words]
                i=[]
                j=[]
                for index in range(len(words)):
                    if index<=window:
                        left_context=[]
                        right_context=[]
                        for word in words[:index]:
                            left_context.append(str(word)+'/l')
                        for word in words[index+1:index+window+1]:
                            right_context.append(str(word)+'/r')
                        context=[]
                        context.append(left_context)
                        context.append(right_context)
                        context=[x for x in context if x!=[]]
                        context=list(itertools.chain.from_iterable(context))
                        
                        target=words[index]
                        for con in context:
                            i.append(vocab.get(target))
                            j.append(vocab.get(con))
                    elif index>window and index<=len(words)-window-1:
                        left_context=[]
                        right_context=[]
                        for word in words[index-2:index]:
                            left_context.append(str(word)+'/l')
                        for word in words[index+1:index+window+1]:
                            right_context.append(str(word)+'/r')

                        context=[]
                        context.append(left_context)
                        context.append(right_context)
                        context=[x for x in context if x!=[]]
                        context=list(itertools.chain.from_iterable(context))
                        
                        target=words[index]
                        for con in context:
                            i.append(vocab.get(target))
                            j.append(vocab.get(con))
                    elif index>len(words)-window-1:
                        left_context=[]
                        right_context=[]
                        for word in words[index-2:index]:
                            left_context.append(str(word)+'/l')
                        for word in words[index+1:]:
                            right_context.append(str(word)+'/r')
                        
                        context=[]
                        context.append(left_context)
                        context.append(right_context)
                        context=[x for x in context if x!=[]]
                        context=list(itertools.chain.from_iterable(context))
                        
                        target=words[index]
                        for con in context:
                            i.append(vocab.get(target))
                            j.append(vocab.get(con))
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
    return cooc_matrix

cooc_matrix_directional=create_dsm_directional('/home/srawat/Documents/UMBC+Wiki/test/test_corpus.txt',vocab_directional,window=2)   
cooc_matrix_directional=cooc_matrix_directional.todok() # dictionary of keys format

## Dependency based cooc matrix
# parent-daughter pairs of every target word=context of the word

# test dep-based cooc matrix 
# dependency parser of spacy doesnt have collapsed dependencies
import spacy
nlp=spacy.load('en_core_web_sm')
test='Bills on ports and immigration were submitted by Senator Brownback, Republican of Kansas'
test2='John hit the ball.'
doc=nlp(test2)
spacy.displacy.serve(doc,style='dep')
for token in doc:
    print(token.text, token.dep_,token.head.text ,[child for child in token.children])
for chunk in doc.noun_chunks:
    print(chunk.text)

# #stanfordnlp implementation
# import stanfordnlp
# # stanfordnlp.download('en')
# nlp=stanfordnlp.Pipeline()
# doc=nlp(test)
# doc.sentences[0].print_tokens()[0]

test='Bills on ports and immigration were submitted by Senator Brownback, Republican of Kansas.'
i=[]
j=[]
test=alt_alpha(test)
doc=nlp(test)
for token in doc:
    print(token.text,token.dep_)
test=alt_alpha(test)
# test=' '.join([word for word in test.split() if word not in stopwords])
doc=nlp(test)
for token in doc:
    print(token.text,token.dep_)
#TODO: fix double arrays slicing
doc=nlp(test)
deps=[]
for token in doc:
    deps.append(token.head.text)
    for child in token.children:
        deps.append(child)
    for dep in deps:
        i.append(token.text)
        j.append(dep)



#TODO: try collapsed dependencies
#! TODO: Add POS filter for tokens (NN, VB, JJ)
import spacy
nlp=spacy.load('en_core_web_sm')
def create_dsm_dep(corpus):
    dep_dsm={}
    t=[]
    c=[]
    with open(corpus,'r') as f:
        for line in f:
            sentences=nltk.sent_tokenize(line)
            for sentence in sentences:
                sentence=alt_alpha(sentence).lower()
                doc=nlp(sentence)
                i=[]
                j=[]
                for token in doc:
                    j.append(token.head.text)
                    for child in token.children:
                        j.append(child)
                    for dep in j:
                        i.append(token.text)
                        j.append(dep)
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
    return cooc_matrix
            
cooc_matrix_dep=create_dsm_dep('/home/srawat/Documents/UMBC+Wiki/test/test_corpus.txt')

# Dependency based DSM with joint context - Chersoni et al.(2016)
# w1,w2 where w1 is the target word and w2=wi#wj where wi is the head of w1
# and wj is the sister node of w1
from scipy.sparse import csr_matrix,dok_matrix

import spacy
nlp=spacy.load('en_core_web_sm',disable=['ner'])

test='Bills on ports and immigration were submitted by Senator Brownback, Republican of Kansas.'
test=alt_alpha(test)
test=re.sub('\s\s+',r' ',test)
doc=nlp(test)
parent=[]
sister=[]
features=[]
for token in doc:
    parent.append(token.head.text)
    for token2 in doc:
        if token2.text==token.head.text:
            for child in token2.children:
                sister.append(child.text+'-'+child.pos_[0].lower()+':'+child.dep_.lower())
                # sister.extend([*[child.text+'-'+child.pos_[0].lower()+':'+child.dep_.lower() for child in token2.children]])
    target=token.text
    for item in sister:
        features.append((target,token.head.text+'-'+token.head.pos_[0].lower()+':'+token.head.dep_.lower()+'#'+item))

#TODO Check effect of lemmatizing contexts joint deps 

# sparse matrices only accept numerical rows and columns
# vocab for joint dep contexts   
import itertools
def create_joint_dep_dsm(corpus):
    joint_dep_dict={}
    nlp=spacy.load('en_core_web_sm',disable=['ner'])
    all_features=[]
    with open(corpus,'r') as f:
        for line in f:
            sentences=nltk.sent_tokenize(line)
            for sentence in sentences:
                sentence=alt_alt_alpha(sentence).lower()
                doc=nlp(sentence)
                sister=[]
                features=[]
                for token in doc:
                    for token2 in doc:
                        if token2.text==token.head.text:
                            for child in token2.children:
                                sister.append(child.text+'-'+child.pos_[0].lower()+':'+child.dep_.lower())
                    target=token.text
                    for item in sister:
                        features.append((target,token.head.text+'-'+token.head.pos_[0].lower()+':'+token.head.dep_.lower()+'#'+item))
                all_features.extend(features)
    joint_dep_dict=dict(Counter(all_features))
    # joint-dep vocab
    joint_vocab={}
    for target,context in joint_dep_dict.keys():
        if target in joint_vocab or context in joint_vocab:
            pass
        elif target not in joint_vocab:
            joint_vocab[target]=len(joint_vocab)
        elif context not in joint_vocab:
            joint_vocab[context]=len(joint_vocab)

    target=[]
    contexts=[]
    for t,c in joint_dep_dict.keys():
        target.append(joint_vocab.get(t))
        contexts.append(joint_vocab.get(c))
    
    frequencies=list(joint_dep_dict.values())
    cooc_matrix_jointdep=csr_matrix((frequencies,(target,contexts)),shape=(len(joint_dep_dict),len(joint_dep_dict)),dtype=np.float64)
    return cooc_matrix_jointdep    

cooc_matrix_joint=create_joint_dep_dsm('/home/srawat/Documents/UMBC+Wiki/test/tiny_corpus.txt')

# test Chersoni joint-dep                            
corpus='/home/srawat/Documents/UMBC+Wiki/test/tiny_corpus.txt'

joint_dep_dict={}
nlp=spacy.load('en_core_web_sm',disable=['ner'])
all_features=[]
with open(corpus,'r') as f:
    for line in f:
        sentences=nltk.sent_tokenize(line)
        for sentence in sentences:
            sentence=alt_alt_alpha(sentence).lower()
            doc=nlp(sentence)
            sister=[]
            features=[]
            for token in doc:
                if token.pos_[0].lower().startswith('n') or token.pos_[0].lower().startswith('v') or token.pos_[0].lower().startswith('j'):
                    for token2 in doc:
                        if token2.text==token.head.text:
                            for child in token2.children:
                                if child.pos_[0].lower().startswith('n') or child.pos_[0].lower().startswith('v') or child.pos_[0].lower().startswith('j'):
                                    sister.append(child.text+'-'+child.pos_[0].lower()+':'+child.dep_.lower())
                    target=token.text
                    for item in sister:
                        features.append((target,token.head.text+'-'+token.head.pos_[0].lower()+':'+token.head.dep_.lower()+'#'+item))
            all_features.extend(features)
joint_dep_dict=dict(Counter(all_features))
# joint-dep vocab
joint_vocab={}
for target,context in joint_dep_dict.keys():
    if target in joint_vocab:
        pass
    elif context in joint_vocab:
        pass
    elif target not in joint_vocab:
        joint_vocab[target]=len(joint_vocab)
    elif context not in joint_vocab:
        joint_vocab[context]=len(joint_vocab)
 
target=[]
contexts=[]
for t,c in joint_dep_dict.keys():
    target.append(joint_vocab.get(t))
    contexts.append(joint_vocab.get(c))

frequencies=list(joint_dep_dict.values())
cooc_matrix_jointdep=csr_matrix((frequencies,(target,contexts)),shape=(len(joint_dep_dict),len(joint_dep_dict)),dtype=np.float64)


# stanfordnlp    
test='Countries of the southern peninsula have renounced the claim for trade freedom.'
import stanfordnlp
nlp=stanfordnlp.Pipeline()
doc=nlp(test)
for sent in doc.sentences:
    for word in sent.words:
        parent=[word2.text for word2 in sent.words if word2.index==word.governor]
        print(word.text,parent)

for sent in doc.sentences:
    for word in sent.words:
        parent=[word2.text for word2 in sent.words if int(word2.index)==word.governor]
        print(word.text,parent[0])

all_features=[]
for sent in doc.sentences:    
    for token in sent.words:
        parent=[word.text+'-'+word.pos[0].lower()+';'+word.dependency_relation.lower() for word in sent.words if int(word.index)==token.governor]
        sister=[]
        features=[]
        if token.pos[0].lower().startswith('n') or token.pos[0].lower().startswith('v') or token.pos[0].lower().startswith('j'):
            for token2 in sent.words:
                if token2.governor==token.governor:
                    sister.append(token2.text+'-'+token2.pos[0].lower()+':'+token2.dependency_relation.lower())
        for item in sister:
            try:
                features.append((token.text,parent[0]+'#'+item))
            except IndexError:
                pass
    all_features.extend(features)


                




                
        





