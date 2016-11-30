import numpy as np
from project.vectorization.embedding import WordEmbedding
from itertools import combinations
from scipy.spatial.distance import euclidean
import gzip


g = WordEmbedding('data/embeddings/glove.6B.50d.txt',
                  verbose=True, use_cache=True, compute_clusters=False)


X = np.array(list(g.vectors.values()))
words_no = len(g.words)

with gzip.open('data/embeddings/glove.distances.50.txt.gz', 'wt') as f:
    li = None
    for (i, Xi), (j, Xj) in combinations(enumerate(X), r=2):
        distance = euclidean(Xi, Xj)
        line = "%d %d %.15f\n" % (i, j, distance)
        f.write(line)
        if li != i:
            li = i
            print("%f%% complete" % (i / words_no))
