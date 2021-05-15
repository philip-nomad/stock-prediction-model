import datetime
import os
import yfinance as yf
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf
import logging

#END_DATE = datetime.date.today() 데이터를 모아야 하기때문에
#START_DATE = END_DATE - relativedelta(years=2)

logging.getLogger('tensorflow').disabled = True

PATH = "./"
os.chdir(PATH)
DIR = 'lstm_score'
if not os.path.exists(DIR):
    os.makedirs(DIR)


# start_date 부터 end_date 까지의 데이터를 가지고 그 다음 날의 종가를 예측합니다.
def start(company_code, END_DATE):
    print(f"company_code: {company_code} lstm 계산 시작")
    START_DATE = END_DATE - relativedelta(years=2)
    # yahoo는 개이기 때문에 +1
    stock = yf.download(company_code+'.KS', start=START_DATE, end=END_DATE + relativedelta(days=1))
    print(stock)

    # 하이퍼파라미터 설정
    input_dcm_cnt = 6  # 입력데이터의 컬럼 개수
    output_dcm_cnt = 1  # 결과데이터의 컬럼 개수
    seq_length = 28  # 1개 시퀸스의 길이(시계열데이터 입력 개수)
    rnn_cell_hidden_dim = 20  # 각 셀의 히든 출력 크기
    forget_bias = 1.0  # 망각편향(기본값 1.0)
    num_stacked_layers = 1  # Stacked LSTM Layers 개수
    keep_prob = 1.0  # Dropout 할때 Keep할 비율
    epoch_num = 1000  # 에포크 횟수 (몇회 반복 학습)
    learning_rate = 0.01  # 학습률

    # 금액&거래량 문자열을 부동소수점형으로 변환함
    stock_info = stock.values[1:].astype(np.float)

    # 데이터 전처리
    # 가격과 거래량 수치의 차이가 많아나서 각각 별도로 정규화한다
    # 가격형태 데이터들을 정규화한다
    # ['Open','High','Low','Close','Adj Close','Volume']에서 'Adj Close'까지 취함
    # 마지막 열 Volume를 제외한 모든 열
    price = stock_info[:,:-1]  # 실제 종가
    on_closing_price = stock['Close'].iloc[-1] # 마지막날의 종가
    norm_price = min_max_scaling(price)
    # norm_price.shape 정규화된 값

    # 거래량형태 데이터를 정규화한다
    # ['Open','High','Low','Close','Adj Close','Volume']에서 마지막 'Volume'만 취함
    volume = stock_info[:, -1:]  # 거래량 volume
    norm_volume = min_max_scaling(volume)  # 거래량형태 데이터 정규화
    # norm_volume.shape

    # 행은 그대로 두고 열을 우측에 붙여 합침
    x = np.concatenate((norm_price, norm_volume), axis=1)  # 세로로 합침
    y = x[:, [-2]]  # 타겟은 주식 종가
    dataX = []  # 입력으로 사용될 Sequence Data
    dataY = []  # 출력(타겟)으로 사용

    for i in range(0, len(y) - seq_length):
        _x = x[i:i + seq_length]
        _y = y[i + seq_length]  # 다음 나타날 주가(정답)
        dataX.append(_x)  # dataX 리스트에 추가
        dataY.append(_y)  # dataY 리스트에 추가

    # 전체의 70% 학습용, 30% 테스트용
    train_size = int(len(dataY) * 0.7)
    test_size = len(dataY) - train_size

    train_size = int(len(dataY) * 0.7)
    test_size = len(dataY) - train_size

    # 학습용 데이터
    trainX = np.array(dataX[0:train_size])
    trainY = np.array(dataY[0:train_size])

    # 데이터를 잘라 테스트용 데이터 생성
    testX = np.array(dataX[train_size:len(dataX)])
    testY = np.array(dataY[train_size:len(dataY)])

    # 텐서플로우 placeholder 생성
    # 입력 X, 출력 Y를 생성
    X = tf.placeholder(tf.float32, [None, seq_length, input_dcm_cnt])
    Y = tf.placeholder(tf.float32, [None, 1])
    print("X:", X)
    print("Y:", Y)

    # 검증용 측정지표를 산출하기 위한 targets, predictions를 생성
    targets = tf.placeholder(tf.float32, [None, 1])
    predictions = tf.placeholder(tf.float32, [None, 1])
    print("targets", targets)
    print("predictions", predictions)

    # num_stacked_layer 개의 층으로 쌓인 Stacked RNNs 생성
    stackedRNNs = [lstm_cell() for _ in range(num_stacked_layers)]  # Stacked LSTM Layers 개수 1
    multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs,
                                              state_is_tuple=True) if num_stacked_layers > 1 else lstm_cell()

    # RNN Cell들을 연결
    hypothesis, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
    # [:. -1]보면 LSTM RNN의 마지막 (hidden)출력만을 사용함
    # 과거 여러 거래일의 주가를 이용해서 다음날의 주가 1개를 예측하기 때문에 Many-to-one 형태
    hypothesis = tf.contrib.layers.fully_connected(hypothesis[:, -1], output_dcm_cnt, activation_fn=tf.identity)
    # hypothesis.shape

    # 손실함수로 평균제곱오차를 사용함.
    loss = tf.reduce_sum(tf.square(hypothesis - Y))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(targets, predictions)))

    train_error_summary = []  # 학습용 데이터의 오류를 중간 중간 기록
    test_error_summary = []  # 테스트용 데이터의 오류를 중간 중간 기록
    test_predict = ''  # 테스트용 데이터로 예측한 결과

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # 학습
    print('학습 시작...')

    for epoch in range(epoch_num):
        _, _loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
        if ((epoch + 1) % 100 == 0) or (epoch == epoch_num - 1):  # 100번째마다 또는 마지막 epoch인 경우
            # 학습용데이터로 rmse오차를 구한다
            train_predict = sess.run(hypothesis, feed_dict={X: trainX})
            train_error = sess.run(rmse, feed_dict={targets: trainY, predictions: train_predict})
            train_error_summary.append(train_error)

            # 테스트용데이터로 rmse오차를 구한다
            test_predict = sess.run(hypothesis, feed_dict={X: testX})
            test_error = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
            test_error_summary.append(test_error)
            # 현재 오류를 출력
            print("epoch: {}, train_error(A): {}, test_error(B): {}, B-A: {}".format(epoch + 1, train_error, test_error,
                                                                                     test_error - train_error))

    # ---------결과 그래프 출력-----------------
    # plt.figure(1)
    # plt.plot(train_error_summary, 'gold')
    # plt.plot(test_error_summary, 'b')
    # plt.xlabel('Epoch(x100)')
    # plt.ylabel('Root Mean Square Error')
    # plt.figure(2)
    # plt.plot(testY, 'r')
    # plt.plot(test_predict, 'b')
    # plt.xlabel('Time Period')
    # plt.ylabel('Stock Price')
    # plt.show()
    # -------------------------------

    # sequence length만큼의 가장 최근 데이터를 슬라이싱 함.
    recent_data = np.array([x[len(x) - seq_length:]])
    # print("recent_data.shape:", recent_data.shape)
    # print("recent_data:", recent_data)

    # 내일의 종가를 예측
    test_predict = sess.run(hypothesis, feed_dict={X: recent_data})
    # print("test_predict", test_predict[0]) 이건 비율
    test_predict = reverse_min_max_scaling(price, test_predict)  # 금액데이터를 역정규화 함
    print("Tomorrow's stock price", test_predict[0])  # 예측한 주가를 출력

    lstm_score = np.round(test_predict[0])
    result = {
        'StockPrediction': lstm_score,
        'ClosingPrice': [on_closing_price],
        'StartDate': [START_DATE],
        'EndDate': [END_DATE]
    }
    lstm_df = pd.DataFrame(result)
    lstm_df.to_csv(f"./{DIR}/{company_code}_{END_DATE}.csv", index=False)

# 정규화
def min_max_scaling(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7)  # 1e-7은 0으로 나누는 오류 예방차

    # 역정규화 : 정규화된 값을 원래의 값으로 되돌림

def reverse_min_max_scaling(org_x, x):  # 종가 예측값
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()

# 모델(LSTM 네트워크) 생성
def lstm_cell(rnn_cell_hidden_dim, forget_bias, keep_prob):
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_cell_hidden_dim,
                                        forget_bias=forget_bias,
                                        state_is_tuple=True,
                                        activation=tf.nn.softsign)
    if keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell