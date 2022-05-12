import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from hmm import *

def read_csv():
    df = pd.read_csv('data/AMZ_data.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df['Close_gap'] = df['Close'].pct_change()
    df['High_gap'] = df['High'].pct_change()
    df['Low_gap'] = df['Low'].pct_change()
    df['Volume_gap'] = df['Volume'].pct_change()
    df['Daily_change'] = (df['Close'] - df['Open']) / df['Open']
    df['Outcome_Next_Day_Direction'] = (df['Volume'].shift(-1) - df['Volume'])

    df = df[1:-1]

    return df

def process_df(df):
    df['Close_gap_LMH'] = pd.qcut(df['Close_gap'], 3, labels=["L", "M", "H"])

    # High_Gap - not used in this example
    df['High_gap_LMH'] = pd.qcut(df['High_gap'], 3, labels=["L", "M", "H"])

    # Low_Gap - not used in this example
    df['Low_gap_LMH'] = pd.qcut(df['Low_gap'], 3, labels=["L", "M", "H"])

    # Volume_Gap
    df['Volume_gap_LMH'] = pd.qcut(df['Volume_gap'], 3, labels=["L", "M", "H"])
    
    # Daily_Change
    df['Daily_change_LMH'] = pd.qcut(df['Daily_change'], 3, labels=["L", "M", "H"])

    # compressed_set = df[abs(df['Outcome_Next_Day_Direction']) > 10000000]
    df['Outcome_Next_Day_Direction'] = np.where((df['Outcome_Next_Day_Direction'] > 0), 1, 0)

    df['Event_Pattern'] = df['Close_gap_LMH'].astype(str) + df['Volume_gap_LMH'].astype(str) + df['Daily_change_LMH'].astype(str)
    new_df = df[['Date', 'Event_Pattern', 'Outcome_Next_Day_Direction']]
    map_bit = {
        'L': 0,
        'M': 1,
        'H': 2
    }

    def convert(s: str):
        assert len(s) == 3
        val = 0
        for i in range(3):
            val += map_bit[s[i]] * (3**i)
        return val
    
    new_df['Event_Pattern'] = new_df['Event_Pattern'].map(lambda x: convert(x))
    new_df = new_df.reset_index()
    return new_df

def build_transition_matrix(df: pd.DataFrame):
    transition_matrix = np.zeros((2, 2), dtype=np.float32)
    for i in range(2):
        cnt = [0 for _ in range(2)]
        for idx in range(len(df)-1):
            if df['Outcome_Next_Day_Direction'][idx] == i:
                cnt[df['Outcome_Next_Day_Direction'][idx+1]] += 1
        for j in range(2):
            transition_matrix[i, j] = cnt[j]/sum(cnt)
    return transition_matrix

def build_emission_matrix(df: pd.DataFrame, limit: int):
    emission_matrix = np.zeros((2, limit), dtype=np.float32)
    for j in range(2):
        cnt = [0 for _ in range(limit)]
        for idx in range(len(df)):
            if df['Outcome_Next_Day_Direction'][idx] == j:
                cnt[df['Event_Pattern'][idx]] += 1
        for i in range(limit):
            emission_matrix[j, i] = cnt[i]/sum(cnt)
    return emission_matrix

if __name__ == '__main__':
    df = read_csv()
    train_df = df[:-500]
    val_df = df[-500:]
    train_df = process_df(train_df)
    val_df = process_df(val_df)
   
    A = build_transition_matrix(train_df)
    B = build_emission_matrix(train_df, 27)
    print(B)
    initial_distribution = np.array((0.5, 0.5))
    
    V = val_df['Event_Pattern'].to_numpy()
    A, B = baum_welch(V, A, B, initial_distribution, 1)
    print(B)
    #predict = viterbi(V, A, B, initial_distribution)
    #label = val_df['Outcome_Next_Day_Direction'].to_numpy()
    #print('Acc:', (predict == label).sum() / len(predict) * 100)