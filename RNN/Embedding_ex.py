import torch
import torch.nn as nn
import random
train_data = 'hong wants to know embedding'
word_set = set(train_data.split())
vocab = {word: i+2 for i, word in enumerate(word_set)}
vocab['<unk>'] = 0
vocab['<pad>'] = 1
print(vocab)

embedding_table = torch.FloatTensor([
                                     [ 0.0,  0.0,  0.0],
                                     [ 0.0,  0.0,  0.0],
                                     [ 0.2,  0.9,  0.3],
                                     [ 0.1,  0.5,  0.7],
                                     [ 0.2,  0.1,  0.8],
                                     [ 0.4,  0.1,  0.1],
                                     [ 0.1,  0.8,  0.9],
                                     [ 0.6,  0.1,  0.1]
                                     ])

sample = 'hong know embedding naver'.split()
indexs = []

for word in sample:
    try:
        indexs.append(vocab[word])
    except:
        indexs.append(vocab['<unk>'])
indexs = torch.LongTensor(indexs)
print(indexs)
lookup_result = embedding_table[indexs,]
print(lookup_result)

########################################################
train_data = 'hong wants to go naver'
word_set = set(train_data.split())
vocab = {word: i+2 for i, word in enumerate(word_set)}
vocab['<unk>'] = 0
vocab['<pad>'] = 1

embedding_layer = nn.Embedding(num_embeddings = len(vocab), embedding_dim = 3, padding_idx = 1)
print(embedding_layer.weight)

