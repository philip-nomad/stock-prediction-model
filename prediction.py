import csv
import os
import warnings

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


import lstm_calculator
import news_contents_sentimental_analysis

warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf

WEIGHT_FOR_LSTM_VALUE = 0.6  # 가중치 a: LSTM 가중치
WEIGHT_FOR_EMOTIONAL_ANALYSIS_VALUE = 0.3  # 가중치 b: 감성분석 점수 가중치
WEIGHT_FOR_PER_VALUE = 0.1  # 가중치 c: PER 점수 가중치

PATH = "./"
os.chdir(PATH)
DIR = 'prediction_score'
STOCKDIR = 'stock'

def mkdir(company_code):
    if not os.path.exists(f"./{DIR}/{company_code}"):
        os.makedirs(f"./{DIR}/{company_code}")

def start(company_code, learning_date):
    learningstart_date=learning_date-relativedelta(days=30)
    for i in range(30):
        with open(f"./{lstm_calculator.DIR}/{company_code}/{company_code}_{learningstart_date}.csv", 'r', -1,
                  'utf-8') as lines:
            next(lines)

            for line in csv.reader(lines):
                lstm_prediction = round(float(line[0]))
                previous_closing_price = round(float(line[1]))

        lstm_value = (lstm_prediction - previous_closing_price) / previous_closing_price  # (다음날 예측 종가 - 오늘 종가) / 오늘 종가
        # 감성분석 값 불러오기
        emotional_analysis_csv = news_contents_sentimental_analysis.calculate_two_weeks(company_code, learning_date)

        # PER 값 불러오기
        company_per_csv = 0
        same_category_per_csv = 0
        with open('./per_data/' + company_code + '.csv', 'r', -1, 'utf-8') as lines:
            next(lines)

            for line in csv.reader(lines):
                company_per_csv = float(line[3])
                same_category_per_csv = float(line[4])

        # 자기 회사 PER 이랑 동일업종 PER 이 모두 양수인 경우에만 per_value 계산
        if company_per_csv > 0 and same_category_per_csv > 0:
            per_value_csv = 1 - company_per_csv / same_category_per_csv  # 1 - 자기 회사 PER / 동일 업종 PER
        else:
            per_value_csv = 0
    

        result = {
            'LearningDate': [learningstart_date],
            'LstmScore': [float(lstm_value)],
            'EmotionalScore': [emotional_analysis_csv],
            'PerScore': [per_value_csv],
            'PreviousClosingPrice': [previous_closing_price],
        }
        prediction_df = pd.DataFrame(result, columns=["LearningDate", "LstmScore", "EmotionalScore", "PerScore",
                                                      "PreviousClosingPrice"])
        if i==0:
            prediction_df.to_csv(f"./{DIR}/{company_code}/{company_code}.csv", index=False, header=True)
        else:
            prediction_df.to_csv(f"./{DIR}/{company_code}/{company_code}.csv", index=False, mode='a', header=False)

        learningstart_date = learningstart_date + relativedelta(days=1)


    stockstart_date=learning_date-relativedelta(days=29)
    stock = pd.read_csv(f"./{STOCKDIR}/{company_code}.KS.csv")

    stock_info = pd.DataFrame(stock)
    stock_info = stock_info.set_index(['Date'])
    stock_info = stock_info.loc[stockstart_date.strftime("%Y-%m-%d"):learning_date.strftime("%Y-%m-%d")]
    stock_info = stock_info.values[0:, 1:].astype(np.float)
    price = stock_info[:, -3]

    xy = pd.read_csv(f"./{DIR}/{company_code}/{company_code}.csv")
    lstm_x = xy.iloc[:, 1]
    emotional_x = xy.iloc[:, 2]
    per_x = xy.iloc[:, 3]
    previous_x = xy.iloc[:, 4]
    today_y = price


    X1 = tf.placeholder(tf.float32, shape=[None])
    X2 = tf.placeholder(tf.float32, shape=[None])
    X3 = tf.placeholder(tf.float32, shape=[None])
    X4 = tf.placeholder(tf.float32, shape=[None])
    Y = tf.placeholder(tf.float32, shape=[None])

    W1 = tf.Variable(0.6, dtype=tf.float32, name='W1', constraint=lambda x: tf.clip_by_value(x, 0, 1))
    W2 = tf.Variable(0.3, dtype=tf.float32, name='W2', constraint=lambda x: tf.clip_by_value(x, 0, 1))
    W3 = 1-(W1+W2)
    sum =W1+W2+W3

    init_op = tf.initialize_all_variables()
    hypothesis = ((X1*W1 + X2*W2 + X3*W3)/100 + 1) * X4
    cost=tf.reduce_mean(tf.square(hypothesis-Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-6)
    train = optimizer.minimize(cost)
    """
    final_W1=0.0
    final_W2=0.0
    final_W3=0.0
    """
    with tf.Session() as sess:
        sess.run(init_op)
        for step in range(5):
            cost_val, hy_val, _ = sess.run(
                [cost, hypothesis, train],
                feed_dict={X1: lstm_x, X2: emotional_x, X3: per_x, X4: previous_x, Y: today_y}
            )
            if step == 4:
                print(step, "\nW3:", sess.run(W3), "\nW2:", sess.run(W2), "\nW1:", sess.run(W1), "\nSum", sess.run(sum))
                """
                final_W1 = sess.run(W1)
                final_W2 = sess.run(W2)
                final_W3 = sess.run(W3)
                """
            else:
                print(step, "Cost", cost_val, "\nPrediction:\n", hy_val, "\nW3:", sess.run(W3), "\nW2:", sess.run(W2), "\nW1:", sess.run(W1),
                    "\nSum", sess.run(sum))

    # print(final_W1,final_W2,final_W3)
    """
    if 부분이 학습을 모두 끝내고 출력하는 거에여 사실 가중치들만 출력하면 되는데 일단 혹시 몰라서 학습내용도 다 출력 시켰습니다.
    final_W1, final_W2, final_W3를 return 하면 최종 가중치들 입니다. 일단 주석 처리 해놓을꼐여 
    이거 return해서 각각 순서대로 lstm, 감성분석, per에 넣어서 계산하면 됩니다.
    """
