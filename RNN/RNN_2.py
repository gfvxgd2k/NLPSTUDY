import torch
import torch.nn as nn
import numpy as np

char_set = ['h','i','e','l','o']

input_size = len(char_set)
hidden_size = len(char_set)
learning_rate = 0.1

x_data = [[0,1,0,2,3,3]]
x_onehot = [[[1,0,0,0,0],
             [0,1,0,0,0],
             [1,0,0,0,0],
             [0,0,1,0,0],
             [0,0,0,1,0],
             [0,0,0,1,0]]]
y_data = [[1,0,2,3,3,4]]

X = torch.FloatTensor(x_onehot)
Y = torch.LongTensor(y_data)

sample = "if you want you"

char_set = list(set(sample)) #중복 문자 제거
char_dic = {c: i for i, c in enumerate(char_set)}

dic_size = len(char_dic)
hidden_size = len(char_dic)
learning_rate = 0.1

sample_idx = [char_dic[c] for c in sample]
x_data = [sample_idx[:-1]]
x_onehot = [np.eye(dic_size)[x] for x in x_data] 
#np.eye
# [100]
# [010] 
# [001]
y_data = [sample_idx[1:]]

X = torch.FloatTensor(x_onehot)
Y = torch.LongTensor(y_data)