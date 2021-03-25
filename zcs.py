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

class Finetune_BERT(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Finetune_BERT, self).__init__()
        



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

def finetune_bert():
    from transformers import BertForSequenceClassification
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    model.train()
    from transformers import AdamW
    optimizer = AdamW(model.parameters(), lr=1e-5)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_batch = ["I love Pixar.", "I don't care for Pixar."]
    encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

def finetune_bert:
    import torch.nn as nn
    bert_model.train()
    def finetune():
        ''' make word embeddings of a given corpus '''
        ''' then dump all the results in .pkl files for later use '''
        for sentence in train_sentences[:size]:
            sentence = sentence[0]
            print(sentence)
            tokenized = tokenizer.tokenize(sentence)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized)
            indexed_tokens = torch.tensor([indexed_tokens])
            segment_ids = [1] * len(tokenized)
            segment_ids = torch.tensor([segment_ids])
            #with torch.no_grad():
            outputs = bert_model(indexed_tokens, segment_ids)
            print(outputs.shape)
            print(outputs.dtype)


    finetune()
    from transformers import BertForSequenceClassification, AdamW, BertConfig

    # Load BertForSequenceClassification, the pretrained BERT model with a single 
    # linear classification layer on top. 
    bert_classi = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    from transformers import BertTokenizer
    input_ids = []
    attention_masks = []
    for sent in train_sentences:
        sent = sent[0]
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = False, # Add '[CLS]' and '[SEP]'
                            max_length = 512,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    MAX_LEN = 100
    for epoch in range(10):
        print(f'======== Epoch {epoch} ========')
        bert_classi.train()
        b_input_ids = input_ids[:MAX_LEN]
        b_input_mask = attention_masks[:MAX_LEN]
        b_labels = labels[:MAX_LEN]
        bert_classi.zero_grad()        
        loss, logits = bert_classi(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)
    #       total_train_loss += loss.item()
        loss.backward()
        print(f'curr loss is {loss}')
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        from torch.utils.data import Dataset
        from torch.utils.data import DataLoader
        class Train_Dataset(Dataset):
            def __init__(self):
                super(Dataset, self).__init__()
            def __getitem__(self, i):
                return train_S1_tensor[i], train_S2_tensor[i], train_labels[i]
            def __len__(self):
                return len(train_labels)
        dataset = Train_Dataset()

        batch_size = 2048

        train_loader = torch.utils.data.DataLoader( dataset, batch_size )


'''+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''
'''

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(sentences, truths, test_size=0.01)
class Trainset():
    def __getitem__(self, i):
        return X_train[i], y_train[i]
    def __len__(self):
        return len(y_train)
class Testset():
    def __getitem__(self, i):
        return X_test[i], y_test[i]
    def __len__(self):
        return len(y_test)
trainset, testset = Trainset(), Testset()

import torch
from torch.utils.data import DataLoader

batch_size = 256
train_loader = torch.utils.data.DataLoader( trainset, batch_size, shuffle=True )
test_loader = torch.utils.data.DataLoader( testset, batch_size, shuffle=True )

preds = []
labels = []

for data, label in test_loader:
    similarity_scores = []
    for sent_pair in data:
        (title, sent) = sent_pair
        input1 = tokenizer(title, return_tensors="pt", padding=True, max_length=512)
        input2 = tokenizer(sent,  return_tensors="pt", padding=True, max_length=512)

        #print(input1)
        #print(input2)

        with torch.no_grad():
            outputs1 = bert_model(**input1)
            outputs2 = bert_model(**input2)
            attention1 = get_attentions(outputs1).detach()
            attention2 = get_attentions(outputs2).detach()
            similarity_scores.append( COS(attention1, attention2).item() )
    score = np.mean(np.array(similarity_scores))
    preds.append(score)
    labels.append(label)
'''
'''+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''
