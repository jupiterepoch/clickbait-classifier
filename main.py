import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn

from models import Attention_Model

import pdb

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
    class Tweet:
        tid = None # tweet id
        title = None
        text = None
        label = None
        def __init__(self, tid, title, text):
            self.tid = tid
            self.title = title
            self.text = text

    truth_file = None
    problem_file = None
    tweets = {}

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
        # postText are always len=1 list
        for t in df:
            self.tweets[t['id']] = Tweet(t['id'], t['targetTitle'].values[0], t['targetParagraphs'].values[0])
        # return df['id'], df['targetTitle'].values[0], df['targetParagraphs'].values[0]

    def preprocessing(self):
        from transformers import BertTokenizer
        for t in self.tweets:
            print(t.title)


def main():
    web17 = Webis17('./data/clickbait17/')
    size = 10
    # TODO: tweet id is a int64, check for overflow!
    (truth_id, label) = web17.get_truths(size)
    web17.preprocessing()
    #(tweet_id, titles, texts) = web17.get_texts(size)
    #print(titles[0])
    #print(texts[0])

if __name__ == '__main__':
    main()