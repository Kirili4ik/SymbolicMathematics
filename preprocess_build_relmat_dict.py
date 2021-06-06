from tqdm import tqdm
import numpy as np
#import jsonlines
import json

from collections import Counter

def _insert(iterable):
    words = []
    for w in iterable:
        words.append(w)
    word_count.update(words)

word_count = Counter()
with open("data/rel_matrix_train_clean.json") as f:
    for line in tqdm(f):
        matrix = json.loads(line)
        new_matrix = [line.split() for line in matrix]
        for tokens in new_matrix:
            for elem in tokens:
                _insert(elem.split("_"))


special_tokens = "unk"
num_spec_tokens = len(special_tokens.split("_"))
# -2 to reserve spots for PAD and UNK token
dict_size = 500
dict_size = dict_size - num_spec_tokens if dict_size and dict_size > num_spec_tokens else dict_size
most_common = word_count.most_common(dict_size)


values = np.array(list(word_count.values()))
keys = np.array(list(word_count.keys()))
idxs = np.argsort(values)[::-1]
vocab = []
for i, (key, value) in enumerate(zip(keys[idxs], values[idxs])):
    print(value, key, i)
    vocab.append(key)
with open("data/rel_vocab.txt", "w") as fout:
    fout.write("\n".join(vocab))
