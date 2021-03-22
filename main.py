import pandas as pd
import numpy as np
import os

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
    truth_file = None
    problem_file = None
    def __init__(self, path):
        self.truth_file = path + 'truth.jsonl'
        self.problem_file = path + 'instances.jsonl'
    def get_truths(self):
        df = pd.read_json(self.truth_file, lines=True)
        print(df)
        return df['', 'truthMean'].to_numpy()

def main():
    web16 = Webis16()
    # web16.get_tweet_content()
    web17 = Webis17('./data/clickbait17/')
    ground_truth = web17.get_truths()
    print(ground_truth.shape)

if __name__ == '__main__':
    main()