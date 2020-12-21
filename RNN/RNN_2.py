import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

""" char_set = ['h','i','e','l','o']

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
 """
sample = "if you want you"
#dictionary
char_set = list(set(sample)) #중복 문자 제거
char_dic = {c: i for i, c in enumerate(char_set)}

input_size = len(char_set)
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

rnn = nn.RNN(input_size, hidden_size, batch_first=True) # batch를 첫 아웃풋으로 아웃함

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), learning_rate) # torch.parameters() 모델의 매개변수, weights, bias

for i in range(100):
    optimizer.zero_grad() # 초기화를 하지않으면 이전 기울기가 지속적으로 축적됨
    outputs, _status = rnn(X)
    loss = criterion(outputs.view(-1, input_size), Y.view(-1))
    loss.backward()
    optimizer.step()
    result = outputs.data.numpy().argmax(axis=2) #index가 2인 곳에서 가장 큰것을 가져옴 
    result_str = ''.join([char_set[c] for c in np.squeeze(result)]) #squeeze 차원이 1인축을 삭제함
    print(i, "loss: ", loss.item(), "prediction: ",result, "true Y: ", y_data, "prediction str: ", result_str)