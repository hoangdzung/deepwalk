from gensim.models import KeyedVectors
import numpy as np 
import sys 

wv_from_txt = KeyedVectors.load_word2vec_format(sys.argv[1], binary=False)  
n_nodes = len(wv_from_txt.vocab)
embeddings = np.zeros((n_nodes, wv_from_txt.vector_size))
for i in range(n_nodes):
        embeddings[i] = wv_from_txt[str(i)]
embeddings = embeddings.astype(np.float32)
np.save(sys.argv[2], embeddings)
