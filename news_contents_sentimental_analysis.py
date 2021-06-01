import csv
import datetime
import os

import pandas as pd

table = dict()

with open('polarity.csv', 'r', -1, 'utf-8') as polarity:
    next(polarity)

    for line in csv.reader(polarity):
        key = str()
        for polarity_word in line[0].split(';'):
            key += polarity_word.split('/')[0]

        table[key] = {'Neg': line[3], 'Neut': line[4], 'Pos': line[6]}

NEWS_SCORE_DIR = 'news_score'
NEWS_SCORE_WORDS_DIR = 'news_score_words'
NEWS_WORDS_DIR = 'news_words'


def mkdir(company_code):
    if not os.path.exists(f"./{NEWS_SCORE_DIR}/{company_code}"):
        os.makedirs(f"./{NEWS_SCORE_DIR}/{company_code}")

    if not os.path.exists(f"./{NEWS_SCORE_WORDS_DIR}/{company_code}"):
        os.makedirs(f"./{NEWS_SCORE_WORDS_DIR}/{company_code}")

    if not os.path.exists(f"./{NEWS_WORDS_DIR}/{company_code}"):
        os.makedirs(f"./{NEWS_WORDS_DIR}/{company_code}")


def get_news_list_by_company_code(company_code, date):
    news_list = []
    try:
        with open(f"./{NEWS_WORDS_DIR}/{company_code}/{company_code}_{str(date)[:10]}.csv", 'r', -1,
                  'utf-8') as news_data:
            next(news_data)

            for news in csv.reader(news_data):
                content = news[2]
                title = news[1]
                news_list.append(content + " " + title)
    except FileNotFoundError:
        return []

    return news_list


def start(company_code, start_date, end_date):
    print(f"company_code: {company_code} 뉴스기사 감성분석 시작")
    mkdir(company_code)

    file_stop_word = open('불용어.txt', 'r', -1, 'utf-8')
    stop_words = file_stop_word.read()
    stop_words = set(stop_words.split('\n'))
    file_stop_word.close()

    while start_date <= end_date:
        negative_list = []
        neutral_list = []
        positive_list = []
        score_word_list = []

        news_list = get_news_list_by_company_code(company_code, start_date)

        for news in news_list:
            word_list = news.split()
            negative = 0
            neutral = 0
            positive = 0
            score_word = ""
            for word in word_list:
                if word in stop_words:
                    continue

                if word in table:
                    score_word += word + " "
                    negative += float(table[word]['Neg'])
                    neutral += float(table[word]['Neut'])
                    positive += float(table[word]['Pos'])

            negative_list.append(negative)
            neutral_list.append(neutral)
            positive_list.append(positive)

            score_word_list.append(score_word)

        negative_list.append(sum(negative_list))
        neutral_list.append(sum(neutral_list))
        positive_list.append(sum(positive_list))

        score_result = {'negative': negative_list, 'neutral': neutral_list, 'positive': positive_list}
        score_df = pd.DataFrame(score_result)
        score_df.to_csv(
            f"./{NEWS_SCORE_DIR}/{company_code}/{company_code}_{str(start_date)[:10]}.csv",
            index=False
        )

        score_word_result = {'words': score_word_list}
        score_word_df = pd.DataFrame(score_word_result)
        score_word_df.to_csv(
            f"./{NEWS_SCORE_WORDS_DIR}/{company_code}/{company_code}_{str(start_date)[:10]}.csv",
            index=False
        )

        start_date += datetime.timedelta(days=1)


def calculate_two_weeks(company_code, target_date):
    neg_sum = 0
    pos_sum = 0
    start_date = target_date
    for diff in range(14):
        day = start_date - datetime.timedelta(days=diff)
        try:
            with open(f"./{NEWS_SCORE_DIR}/{company_code}/{company_code}_{str(day)[:10]}.csv",
                      'r',
                      -1,
                      'utf-8') as num_data:
                next(num_data)
                pos_list = []
                neg_list = []
                for nums in csv.reader(num_data):
                    pos_list.append(float(nums[2]))
                    neg_list.append(float(nums[0]))
            neg_sum += neg_list[-1]
            pos_sum += pos_list[-1]

        except FileNotFoundError:
            # 기사가 아예 없는 날은 파일이 생성이 안됨
            # 기사가 있는데 전처리 후 단어가 없거나, 불용어만 있어서 점수를 계산할 수 없는 경우, pos, neu, neg 모두 0
            continue
    if pos_sum + neg_sum == 0:
        portion = 0
    else:
        ratio = pos_sum / (pos_sum + neg_sum)
        portion = (ratio / 0.53) - 1

    return portion


def get_score_word(company_code, date):
    news_list = ""
    try:
        with open(f"./{NEWS_SCORE_WORDS_DIR}/" + company_code + "/" + company_code + "_" + str(date)[:10] + '.csv',
                  'r', -1,
                  'utf-8') as news_data:
            next(news_data)

            for news in csv.reader(news_data):
                news_list += news[0] + " "
    except FileNotFoundError:
        return ""

    news_list = news_list.strip()
    return news_list


def get_sentimental_score(company_code, date):
    neg = []
    neu = []
    pos = []

    try:
        with open(f"./{NEWS_SCORE_DIR}/" + company_code + "/" + company_code + "_" + str(date)[:10] + '.csv',
                  'r', -1,
                  'utf-8') as news_data:
            next(news_data)
            for news in csv.reader(news_data):
                neg.append(float(news[0]))
                neu.append(float(news[1]))
                pos.append(float(news[2]))

    except FileNotFoundError:
        return [0, 0, 0]

    return [neg[-1], neu[-1], pos[-1]]


def get_lstm_prediction_data(company_code, date):
    # LSTM 값 불러오기
    with open(f"./lstm_score/{company_code}/{company_code}_{date}.csv", 'r', -1, 'utf-8') as lines:
        next(lines)

        for line in csv.reader(lines):
            lstm_prediction_price = round(float(line[0]))
            closing_price = int(float(line[1]))

    return lstm_prediction_price, closing_price
