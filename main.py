import pandas as pd
import numpy as np
import os

import logging

from models import Attention_Model
from models import Hugging_Model

import pdb

#logger = logging.basicConfig(level=info)

class Webis16:
    def __init__(self):
        self.truth_path = './data/webis-clickbait-16/truth/'
        self.problem_path = './data/webis-clickbait-16/problems/'

    def get_truths(self, annotator="majority"):
        '''
        reads in the ground truth file and returns dictionary `ground_truth`
        `ground_truth` uses tweet-id as key and 0/1 as value
        0 indicates not clickbait, 1 indicates tweet is clickbait
        '''
        truth_file_path = self.truth_path+annotator+".csv"
        if not os.path.exists(truth_file_path):
            print(f"Ground truth {truth_file_path} does not exist!")
            exit(1)
        df = pd.read_csv(truth_file_path, header=None)
        df.columns = ['tweet_id', 'clickbaitness']
        df = df.replace({ 'clickbaitness' : {'no-clickbait':0, 'clickbait':1} })
        ground_truth = df.set_index('tweet_id').to_dict()['clickbaitness']
        return ground_truth

    def get_tweet_content(self, filename='607668877594497024/607668877594497024.html'):
        file = self.problem_path + filename
        from bs4 import BeautifulSoup
        f = open(file, 'r')
        soup = BeautifulSoup(f.read())
        print(soup.prettify())
        f.close()

class Webis17:
    truth_file = None
    problem_file = None
    corpus = [] # (title, paragraphs, label)

    def __init__(self, path):
        self.truth_file = path + 'truth.jsonl'
        self.problem_file = path + 'instances.jsonl'

    def get_truths(self, size=100):
        df = pd.read_json(self.truth_file, lines=True)
        df = df.loc[:size, :]
        return df['id'], df['truthMean'].values

    def get_texts(self, size=100):
        df = pd.read_json(self.problem_file, lines=True)
        df = df.loc[:size, :]
        return df['id'], df['targetTitle'], df['targetParagraphs']

    def build_corpus(self, size=100):
        (truth_id, label) = self.get_truths(size)
        ground_truth = {truth_id[i] : label[i] for i in range(len(label))}
        (tweet_id, titles, texts) = self.get_texts(size)
        for i, tid in enumerate(tweet_id):
            try:
                self.corpus.append( (titles[i], ' '.join(txt for txt in texts[i]), ground_truth[tid]) ) # tid is discarded from now on
            except KeyError:
                pass
                #print(f'Tweet {tid} is not in ground truth!')
        print(f'Getting {len(self.corpus)} valid examples from training set.')

class Trainset():
    #from torch.utils.data import Dataset
    #from torch.utils.data import DataLoader
    def __init__(self):
        super(Trainset, self).__init__()
    def __getitem__(self, i):
        title, text, label = web17.corpus[i]
        return (title, text, label)
    def __len__(self):
        return len(web17.corpus)

web17 = Webis17('./data/clickbait17/')
trainset = Trainset()

def get_attentions(outputs, layer=0, attention_head=0, avg=False):
    '''
    get the particular output for a particular layer and attention head
    layer -> 0 to 11
    attention_head -> 0 to 11
    '''
    if avg:
        #avg over all attention heads in a layer
        return outputs[layer].squeeze(0).mean(dim=0)  
    #return values for a particular attention head inside a specific layer
    return outputs[layer].squeeze(0)[attention_head]


def demo():
    import torch
    import torch.nn as nn
    import numpy as np
    from transformers import AutoTokenizer, AutoModel
    COS = torch.nn.CosineSimilarity(dim=0)


    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = AutoModel.from_pretrained("bert-base-uncased")
    batch_size = 32
    train_loader = torch.utils.data.DataLoader( trainset, batch_size )
    
    truths = []
    preds = []
    # for (title, text, label) in train_loader:
    for tweet in web17.corpus:
        # fixme: using only first 512 to bypass an error
        title, text, label = tweet[0], tweet[1][:512], tweet[2]
        #print('@',title, text, '@')
        truths.append(label)
        input1 = tokenizer(title, return_tensors="pt")
        input2 = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs1 = bert_model(**input1)
            outputs2 = bert_model(**input2)
            attention1 = get_attentions(outputs1).detach()
            attention2 = get_attentions(outputs2).detach()
            preds.append( COS(attention1, attention2).item() )
        #print(attention1.shape, attention2.shape)
        #print( COS(attention1, attention2) )
    #print(preds[0])
    #loss = torch.nn.MSELoss()
    #print(f'Loss for naive similarity model is {loss(np.array(preds), np.array(truths))}')
    import json
    with open('preds.json', 'w') as fout:
        json.dump ( preds, fout, indent=4 )
    with open('truths.json', 'w') as fout:
        json.dump ( truths, fout, indent=4 )

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


def main():
    web17.build_corpus(size=19538)
    #demo()

if __name__ == '__main__':
    main()