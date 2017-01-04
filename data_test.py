from project.augmentation import CombinerAugmenter
from project.data import Dataset
from project.vectorization.embedding import WordEmbedding


g = WordEmbedding('data/embeddings/glove.6B.50d.txt',
                  verbose=True, use_cache=True,
                  compute_clusters=True)

#t = Dataset('data/prepared/trec.txt.gz')
t = Dataset(word_embedding=g,
            filename='data/prepared/trec.txt.gz',
            augmenter=CombinerAugmenter(max_window_size=3))


a, b, c = t.get_prepared_data(train_data_percentage=.70)

print("Computing...")
a = list(a)
b = list(b)
c = list(c)
print("Finished.")

for x in [a, b, c]:
    for xq, xa in x:
        print(xq.shape, xa)


print(len(a), len(b), len(c))