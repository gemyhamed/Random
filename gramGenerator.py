import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from csv import DictReader, DictWriter
import argparse


def get_data(path,field):
    df = pd.read_csv(path,usecols=[field])
    return df[field]
    

def vectorizer(df):
    
    cnt = CountVectorizer(ngram_range=(2,3),token_pattern = r"(?u)\b[\\]?\w\w+\b")
    corpus = df.values.tolist()
    x  = cnt.fit_transform(corpus)
    key = cnt.get_feature_names()
    val = np.sum(x,axis=0).tolist()[0]
    dic = dict(zip(key,val))
    rep = pd.Series(dic)
    return rep


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='input csv file')
    parser.add_argument('--output_file', help='output folder path', default='Ngrams.csv')
    args = parser.parse_args()
    
    vectorizer(get_data(args.input_file,'title')).reset_index().\
    rename({'index':'gram',0:'count'},axis=1).set_index('gram').to_csv(args.output_file)
    
    