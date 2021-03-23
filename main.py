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

    def get_truths(self, size=-1):
        df = pd.read_json(self.truth_file, lines=True)
        df = df.loc[:size, :]
        print(df)
        return df['id'], df['truthMean'].values

    def get_texts(self, size=-1):
        df = pd.read_json(self.problem_file, lines=True)
        df = df.loc[:size, :]
        print(df)
        print(df.columns)
        return df['id'], df['targetTitle'], df['targetParagraphs']

    def build_corpus(self):
        size = 10
        # TODO: tweet id is a int64, check for overflow!
        (truth_id, label) = self.get_truths(size)
        ground_truth = {truth_id[i] : label[i] for i in range(len(label))}
        (tweet_id, titles, texts) = self.get_texts(size)
        for i, tid in enumerate(tweet_id):
            try:
                self.corpus.append( (titles[i], ' '.join(txt for txt in texts[i]), ground_truth[tid]) ) # tid is discarded from now on
            except KeyError:
                print(f'Tweet {tid} is not in ground truth!')

class Trainset():
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    def __init__(self):
        super(Dataset, self).__init__()
    def __getitem__(self, i):
        title, text, label = web17.corpus[i]
        return (title, text, label)
    def __len__(self):
        return len(web17.corpus)

web17 = Webis17('./data/clickbait17/')
trainset = Trainset()


def demo():
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    COS = torch.nn.CosineSimilarity()


    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = AutoModel.from_pretrained("bert-base-uncased")
    batch_size = 32
    train_loader = torch.utils.data.DataLoader( trainset, batch_size )
    
    truths = []
    preds = []
    for title, text, label in train_loader:
        truths.append(label)
        input1 = tokenizer(title)
        input2 = tokenizer(text)
        outputs1 = bert_model(title)
        outputs2 = bert_model(text)
        preds.append(COS(outputs1, outputs2))

    loss = torch.nn.MSELoss()
    print(f'Loss for naive similarity model is {loss(preds, truths)}')



def main():
    web17.build_corpus()
    demo()

if __name__ == '__main__':
    main()