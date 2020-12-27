#출저: https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/code_practices/Sequence_to_Sequence_with_LSTM_Tutorial.ipynb
from torch.nn.modules import dropout
import spacy
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k
import torch
import torch.nn as nn



spacy_en = spacy.load('en')
spacy_de = spacy.load('de')
BATCH_SIZE = 128
#토큰화
def main():
    
    #토큰화 기능 확인
    tokenized = spacy_en.tokenizer('I am a graduate student.')
    for i, token in enumerate(tokenized):
        print('index {}: {}'.format(i,token))

    SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='eos', lower=True)
    TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='eos', lower=True)

    train_dataset, valid_dataset, test_dataset = Multi30k.splits(exts=('.de','.en'), fields=(SRC, TRG))

    print(f"학습 데이터셋(training dataset) 크기: {len(train_dataset.examples)}개")
    print(f"평가 데이터셋(validation dataset) 크기: {len(valid_dataset.examples)}개")
    print(f"테스트 데이터셋(testing dataset) 크기: {len(test_dataset.examples)}개")

    #dict 생성 빈도수 2
    SRC.build_vocab(train_dataset, min_freq=2)
    TRG.build_vocab(train_dataset, min_freq=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #한 배치에 포함된 문장의 단어수를 유사하게 만들어주는게 좋음
    #BucketIterator를 이용함
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_dataset, valid_dataset, test_dataset),
        batch_size = BATCH_SIZE,
        device = device
    )

    for i, batch in enumerate(train_iterator):
        src = batch.src
        trg = batch.trg

        





def tokenize_de(text):
    return [token.text for token in spacy_de.tokenizer(text)][::-1]
def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, n_layers, dropout_ratio):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embed_dim)
        #lstm
        self.hidden_Dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout_ratio)
        self.dropout = nn.Dropout(dropout_ratio)

    def fowrard(self, src):

        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell)= self.rnn(embedded)

        return hidden, cell

if __name__ == '__main__':
    main()
