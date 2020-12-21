#pred long sentence with sliding window
from typing import Sequence
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sentence = ("node js, operating systems unity andengine python visual studio code edit file view go terminal if you want i dont know how to get this long long sentence")
char_set = list(set(sentence))
char_dic = {c: i for i,c in enumerate(char_set)}
sequence_length = 10

dic_size = len(char_dic)
hidden_size = len(char_dic)
lr =0.01

x_data = []
y_data = []


for i in range(0, len(sentence)-sequence_length):
    x_str = sentence[i:i+sequence_length]
    y_str = sentence[i+1:i+sequence_length+1]
    print(i, x_str,'>',y_str)
    x_data.append([char_dic[c] for c in x_str])
    y_data.append([char_dic[c] for c in y_str])

x_onehot = [np.eye(dic_size)[x] for x in x_data]

X = torch.FloatTensor(x_onehot)
Y = torch.LongTensor(y_data)

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(Net, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim,hidden_dim,bias=True)

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x

net = Net(dic_size, hidden_size, 2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr)

for i in range(100):
    optimizer.zero_grad()
    outputs = net(X)
    loss = criterion(outputs.view(-1,dic_size), Y.view(-1))
    loss.backward()
    optimizer.step()

    resutls = outputs.argmax(dim =2)
    predict_str = ""
    for j, result in enumerate(resutls):
        print(i, j, ''.join([char_set[t] for t in result]), loss.item())
        if j == 0:
            predict_str += ''.join([char_set[t] for t in result])
        else:
            predict_str += char_set[result[-1]]