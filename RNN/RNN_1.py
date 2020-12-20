import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np

n_hidden = 45
lr = 0.001
epochs = 1000

string = "hi my name is kyuhong! nice meet u"
chars = "abcdefghijklmnopqrstuvwxyz ?!.:;01"
char_list = [i for i in chars]
n_letters = len(char_list)

def string_to_onehot(string):
    start = np.zeros(shape = len(char_list), dtype = int)
    end = np.zeros(shape = len(char_list), dtype = int)
    start[-2] = 1
    end[-1] = 1
    for i in string:
        idx = char_list.index(i) #index() is find value minimum index
        zero = np.zeros(shape = n_letters, dtype = int)
        zero[idx] = 1
        start = np.vstack([start, zero]) #add 행
    output = np.vstack([start, end])
    return output

def onehot_toword(onehot_1):
    onehot = torch.Tensor.numpy(onehot_1)
    return char_list[onehot.argmax()] #argmax maximum value index 즉 1의 index

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.i2o = nn.Linear(hidden_size, output_size)
        self.act_fn = nn.Tanh()

    def forward(self, input, hidden):
        hidden = self.act_fn(self.i2h(input) + self.h2h(hidden))
        output = self.i2o(hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

rnn = RNN(n_letters, n_hidden, n_letters)
loss_func = nn.MSELoss() #두개의 평균제곱오차 즉, 두개의 거리차이
optimizer = torch.optim.Adam(rnn.parameters(), lr = lr) # torch.parameters() 모델의 매개변수, weights, bias

one_hot = torch.from_numpy(string_to_onehot(string)).type_as(torch.FloatTensor()) # Tensor()와 다르게 numpy array도 변경됨
print(one_hot)
#train
for i in range(epochs):
    rnn.zero_grad() #init grad 0
    total_loss = 0
    hidden = rnn.init_hidden() #학습을 위해 0으로 초기화

    for j in range(one_hot.size()[0] - 1):
        input_ = one_hot[j:j+1,:]
        target = one_hot[j+1]

        output, hidden = rnn.forward(input_, hidden)
        loss = loss_func(output.view(-1), target.view(-1)) # view() reshape
        total_loss += loss
        input_ = output

    total_loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(i, total_loss)

start = torch.zeros(1,len(char_list))
start[:-2] = 1

#test
with torch.no_grad():
    hidden = rnn.init_hidden()
    input_ = start
    output_string = ""
    for i in range(len(string)):
        output, hidden = rnn.forward(input_, hidden)
        output_string += onehot_toword(output.data)
        input_ = output
print(output_string)