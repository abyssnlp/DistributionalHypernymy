 # Target and context to freq dict
import itertools
from collections import Counter
from scipy.sparse import csr_matrix,dok_matrix,save_npz
import pickle

def gen(filee):
    with open(filee,'r') as f:
        for line in f:
            yield int(line.strip())

num=gen('/home/srawat/Documents/UMBC+Wiki/dsm_files/test.txt')            
try:
    next(num)
except StopIteration:
    pass


target=gen('/home/srawat/Documents/UMBC+Wiki/dsm_files/target.txt')
context=gen('/home/srawat/Documents/UMBC+Wiki/dsm_files/context.txt')
with open('/home/srawat/Documents/UMBC+Wiki/dsm_files/tuples.txt','w') as g:
    for t,c in zip(target,context):
        g.write(str(t)+'\t'+str(c)+'\n')

# generator based hash table
dict_file=gen('/home/srawat/Documents/UMBC+Wiki/dsm_files/tuples.txt')
for line in dict_file:
    target,context=line.split('\t')
    target,context=int(target),int(context)
    if (target,context) in count_dict:
        count_dict[(target,context)]+=1
    else:
        count_dict[(target,context)]=1

# dict chunk by chunk

from dask.distributed import Client
client=Client(n_workers=5,threads_per_worker=2,processes=False,memory_limit='2GB')
import dask.dataframe as dd

pair_data=dd.read_table('/home/srawat/Documents/UMBC+Wiki/dsm_files/tuples.txt')
pair_data.rename(columns={2822:'target',80:'context'})



# Tuple to dict
import pickle
from tqdm import tqdm
count_dict=dict()
with open('/home/srawat/Documents/UMBC+Wiki/dsm_files/tuples.txt','r') as g:
    for line in tqdm(g):
        target,context=line.split('\t')
        target=int(target)
        context=int(context)
        if (target,context) in count_dict:
            count_dict[(target,context)]+=1
        else:
            count_dict[(target,context)]=1
with open('/home/srawat/Documents/UMBC+Wiki/dsm_files/count_dict.pkl','wb') as h:
    pickle.dump(count_dict,h,protocol=pickle.HIGHEST_PROTOCOL)        



test=[]
with open('/home/srawat/Documents/UMBC+Wiki/dsm_files/test.txt','r') as f:
    for line in f:
        test.append(int(line.strip()))


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
save_npz('/home/srawat/Documents/Distributional Hypernymy/window_dsm_csr.npz',cooc_matrix)
# save_npz('/home/srawat/Documents/UMBC+Wiki/test/test_dsm.npz',cooc_matrix)