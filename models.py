import torch
import torch.nn as nn

class Attention_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers): # , batch_size
        super(Attention_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.dense = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.linear = nn.Linear(hidden_size, 1)
        self.cos = nn.CosineSimilarity()
        
    def forward(self, input1, input2, init_hc):
        # input 1 and input 2 are already embeddings
        out1, (hn1, cn1) = self.rnn(input1, init_hc)
        out2, (hn2, cn2) = self.rnn(input2, init_hc)
        out1 = self.linear(hn1).squeeze().transpose(0, 1)
        out2 = self.linear(hn2).squeeze().transpose(0, 1)
        #out1 = self.dense(input1).squeeze()
        #out2 = self.dense(input2).squeeze()
        return 1 + 4 * self.cos(out1, out2)

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


class Hugging_Model():
    def play():
        pass

''' junk yard below

class Tweet:
        tid = None # tweet id
        title = None
        text = None
        label = None
        def __init__(self, tid, title, text):
            self.tid = tid
            self.title = title
            self.text = text

'''