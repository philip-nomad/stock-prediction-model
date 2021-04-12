import requests
import re
import os
import pandas as pd
import csv
import io
import datetime

table=dict()

with open('polarity.csv', 'r', -1, 'utf-8') as polarity:
    next(polarity)

    for line in csv.reader(polarity):
        key = str()
        for word in line[0].split(';'):
            key += word.split('/')[0]

        table[key] = {'Neg': line[3], 'Neut': line[4], 'Pos': line[6]}

def get_text(code):
    word_list = []
    with open('./words/'+code+'.csv', 'r', -1, 'utf-8') as news:
        next(news)

        for line in csv.reader(news):
            words = line[2]
            words += line[1]
            word_list.append(words)
    return word_list

def text_processing(code):
    word_list = get_text(code)
    f = open("./score/"+code+'.csv', "w+")
    f.close()
    file_stop_word=open('불용어.txt','r',-1,'utf-8')
    stop_words=file_stop_word.read()
    stop_word_list=[]
    negative_list = []
    neutral_list = []
    positive_list = []
    for word in stop_words.split('\n'):
        if word not in stop_word_list:
            stop_word_list.append(word)
    file_stop_word.close()

    for words in word_list:
        list = words.split()
        negative = 0
        neutral = 0
        positive = 0
        for word in list:
            if word in table:
                negative += float(table[word]['Neg'])
                neutral += float(table[word]['Neut'])
                positive += float(table[word]['Pos'])
        negative_list.append(negative)
        neutral_list.append(neutral)
        positive_list.append(positive)

    columns = ['negative', 'neutral', 'positive']
    df = pd.DataFrame(columns=columns)
    df['negative']=negative_list
    df['neutral']=neutral_list
    df['positive']=positive_list

    df.to_csv("./score/"+code+'.csv', index=False)
    ds = pd.read_csv("./score/"+code+'.csv')