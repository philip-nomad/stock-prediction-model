import datetime
import os

import FinanceDataReader
import numpy as np
import pandas as pd
import tensorflow as tf
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM, Conv1D
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

END_DATE = datetime.date.today()
START_DATE = END_DATE - relativedelta(years=1)
WINDOW_SIZE = 5
BATCH_SIZE = 32

PATH = "./"
os.chdir(PATH)
DIR = 'lstm_score'
if not os.path.exists(DIR):
    os.makedirs(DIR)


# start_date 부터 end_date 까지의 데이터를 가지고 그 다음 날의 종가를 예측합니다.
def start(company_code):
    print(f"company_code: {company_code} lstm 계산 시작")
    stock = FinanceDataReader.DataReader(company_code, START_DATE, END_DATE)
    ori_closing_price = stock[['Close']]
    stock.to_csv('stock.csv')

    scaler = MinMaxScaler()
    scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    scaled = scaler.fit_transform(stock[scale_cols])
    df = pd.DataFrame(scaled, columns=scale_cols)

    x_train, x_test, y_train, y_test = train_test_split(df.drop('Close', 1),
                                                        df['Close'],
                                                        test_size=0.2,
                                                        random_state=0,
                                                        shuffle=False)

    # train_data 는 학습용 데이터셋, test_data 는 검증용 데이터셋 입니다.
    train_data = window_dataset(y_train, True)
    test_data = window_dataset(y_test, False)

    # 아래의 코드로 데이터셋의 구성을 확인해 볼 수 있습니다.
    # X: (batch_size, window_size, feature)
    # Y: (batch_size, feature)
    for data in train_data.take(1):
        print(f'데이터셋(X) 구성(batch_size, window_size, feature 갯수): {data[0].shape}')
        print(f'데이터셋(Y) 구성(batch_size, window_size, feature  갯수): {data[1].shape}')

    model = Sequential([
        # 1차원 feature map 생성
        Conv1D(filters=32, kernel_size=5,
               padding="causal",
               activation="relu",
               input_shape=[WINDOW_SIZE, 1]),
        # LSTM
        LSTM(16, activation='tanh'),
        Dense(16, activation="relu"),
        Dense(1),
    ])

    # Sequence 학습에 비교적 좋은 퍼포먼스를 내는 Huber()를 사용합니다.
    Huber()
    optimizer = Adam(0.0005)
    model.compile(loss=Huber(), optimizer=optimizer, metrics=['mse'])

    # early_stopping 은 10번 epoch 동안 val_loss 개선이 없다면 학습을 멈춥니다.
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    # val_loss 기준 체크포인터도 생성합니다.

    model.fit(train_data, validation_data=test_data, epochs=50)
    pred = model.predict(test_data)

    pred = reverse_min_max_scaling(ori_closing_price, pred)  # 역정규화
    print(pred[0])
    print(pred)

    lstm_score = round(pred[-1][0])
    result = {
        'StockPrediction': [lstm_score],
        'ClosingPrice': [ori_closing_price],
        'StartDate': [START_DATE],
        'EndDate': [END_DATE]
    }

    lstm_df = pd.DataFrame(result)
    lstm_df.to_csv(f"./{DIR}/{company_code}_{END_DATE}.csv", index=False)


def window_dataset(series, shuffle):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(WINDOW_SIZE + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(WINDOW_SIZE + 1))
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.map(lambda w: (w[:-1], w[-1]))

    return ds.batch(BATCH_SIZE).prefetch(1)


# 역정규화 : 정규화된 값을 원래의 값으로 되돌림
def reverse_min_max_scaling(org_x, x):  # 종가 예측값
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)

    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()
