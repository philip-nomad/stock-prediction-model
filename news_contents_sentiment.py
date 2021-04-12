import csv

import pandas as pd

table = dict()

with open('polarity.csv', 'r', -1, 'utf-8') as polarity:
    next(polarity)

    for line in csv.reader(polarity):
        key = str()
        for polarity_word in line[0].split(';'):
            key += polarity_word.split('/')[0]

        table[key] = {'Neg': line[3], 'Neut': line[4], 'Pos': line[6]}


def get_news_list_by_company_code(company_code):
    news_list = []
    with open('./words/' + company_code + '.csv', 'r', -1, 'utf-8') as news_data:
        next(news_data)

        for news in csv.reader(news_data):
            content = news[2]
            title = news[1]
            news_list.append(content + " " + title)

    return news_list


def text_processing(company_code):
    f = open("./score/" + company_code + '.csv', "w+")
    f.close()

    file_stop_word = open('불용어.txt', 'r', -1, 'utf-8')
    stop_words = file_stop_word.read()
    stop_words = set(stop_words.split('\n'))
    file_stop_word.close()

    negative_list = []
    neutral_list = []
    positive_list = []

    news_list = get_news_list_by_company_code(company_code)
    for news in news_list:
        word_list = news.split()
        negative = 0
        neutral = 0
        positive = 0
        for word in word_list:
            if word in stop_words:
                continue

            if word in table:
                negative += float(table[word]['Neg'])
                neutral += float(table[word]['Neut'])
                positive += float(table[word]['Pos'])

        negative_list.append(negative)
        neutral_list.append(neutral)
        positive_list.append(positive)

    columns = ['negative', 'neutral', 'positive']
    df = pd.DataFrame(columns=columns)
    df['negative'] = negative_list
    df['neutral'] = neutral_list
    df['positive'] = positive_list

    df.to_csv("./score/" + company_code + '.csv', index=False)
    pd.read_csv("./score/" + company_code + '.csv')
