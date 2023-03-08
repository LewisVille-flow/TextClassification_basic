import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

#### GRU
class GRUModel(nn.Module):
    '''
    input_size -> text vocab size
    '''
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim, num_layers, batch_first):
        super(GRUModel, self).__init__()

        self.num_layers = num_layers
        self.batch_first = batch_first   
        self.hidden_dim = hidden_dim

        """
        TODO: Implement your own model. You can change the model architecture.
        """
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.drop = nn.Dropout(0.5);
        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=0.5,
                                bidirectional=False)
        self.fc = nn.Linear(hidden_dim, output_size)

    # the size of x in forward is (seq_length, batch_size) if batch_first=False
    def forward(self, x):
        batch_size = x.size(0) if self.batch_first else x.size(1)

        #h_0: (num_layers * num_directions, batch_size, hidden_size)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)

        embedding = self.embedding(x)
        embedding = self.drop(embedding)

        outputs, hidden = self.rnn(embedding)  # outputs.shape -> (sequence length, batch size, hidden size)

        h_n = hidden[0]
        #outputs = outputs[:, -1, :] if self.batch_first else outputs[-1, :, :]
        
        output = self.fc(h_n)
        
        return output, hidden

### LSTM
class LSTMModel(nn.Module):
    '''
    input_size -> text vocab size
    '''
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim, num_layers, batch_first):
        super(LSTMModel, self).__init__()

        self.num_layers = num_layers
        self.batch_first = batch_first   
        self.hidden_dim = hidden_dim

        """
        TODO: Implement your own model. You can change the model architecture.
        """
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.drop = nn.Dropout(0.5);
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=0.5,
                                bidirectional=False)
        self.fc = nn.Linear(hidden_dim, output_size)
        #self.fc2 = nn.Linear(32, output_size)

    # the size of x in forward is (seq_length, batch_size) if batch_first=False
    def forward(self, x):
        batch_size = x.size(0) if self.batch_first else x.size(1)

        #h_0: (num_layers * num_directions, batch_size, hidden_size)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)

        embedding = self.embedding(x)
        embedding = self.drop(embedding)

        outputs, hidden = self.rnn(embedding)
        # lstm returns output, (h_n, c_n)
        # if batched input, h_n = (num_layers, batch_size, hidden_size)
        # here, fc layer needs hidden size(== hidden[0][-1])
        

        h_n = hidden[0]
        # outputs = outputs[:, -1, :] if self.batch_first else outputs[-1, :, :]
        
        #o1 = self.fc1(h_n)
        output = self.fc(h_n)
        
        return output, hidden

### with pacekd_sequence
class PackedLSTMModel(nn.Module):
    '''
    input_size -> text vocab size
    '''
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim, num_layers, batch_first):
        super(PackedLSTMModel, self).__init__()

        self.num_layers = num_layers
        self.batch_first = batch_first   
        self.hidden_dim = hidden_dim

        """
        TODO: Implement your own model. You can change the model architecture.
        """
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.drop = nn.Dropout(0.5);
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=0.5,
                                bidirectional=False)
        self.fc1 = nn.Linear(hidden_dim, output_size)

    # the size of x in forward is (seq_length, batch_size) if batch_first=False
    def forward(self, x, input_length):

        embedding = self.embedding(x)
        packed_input = pack_padded_sequence(embedding, input_length.tolist())
        packed_output, hidden = self.rnn(packed_input)
        
        h_n = hidden[-1]
        # packed로 인해서, RNN의 인풋은 batched가 아닐 것같다. 그래서 일단, [0] == hidden state로 가정하고 해보자.
        # 다시 생각해봤을때, RNN h_n은 tensor of shape (D * num\_layers, H_{out}) 이니까, 1로 하는게 맞는거같아 다시 시도. 
        # 별 차이는 없다.

        out = self.fc1(h_n)
        
        return out
