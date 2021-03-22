import pandas as pd
import numpy as np
import os

import pdb

truth_path = './data/webis-clickbait-16/truth/'
problem_path = './data/webis-clickbait-16/problems/'


def get_truths(annotator="majority"):
    '''
    reads in the ground truth file and returns dictionary `ground_truth`
    `ground_truth` uses tweet-id as key and 0/1 as value
    0 indicates not clickbait, 1 indicates tweet is clickbait
    '''

    truth_file_path = truth_path+annotator+".csv"
    if not os.path.exists(truth_file_path):
        print(f"Ground truth {truth_file_path} does not exist!")
        exit(1)
    df = pd.read_csv(truth_file_path, header=None)
    df.columns = ['tweet_id', 'clickbaitness']
    df = df.replace({ 'clickbaitness' : {'no-clickbait':0, 'clickbait':1} })
    ground_truth = df.set_index('tweet_id').to_dict()['clickbaitness']
    return ground_truth

def get_tweet_content(filename=problem_path+'607668877594497024/607668877594497024.html'):
    from bs4 import BeautifulSoup
    f = open(filename, 'r')
    soup = BeautifulSoup(f.read())
    print(soup.prettify())
    f.close()


def main():
    #ground_truth = get_truths()
    get_tweet_content()

if __name__ == '__main__':
    main()