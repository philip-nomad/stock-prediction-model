import os

import FinanceDataReader
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, LSTM, Conv1D
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

STOCK_CODE = '005930'
stock = FinanceDataReader.DataReader(STOCK_CODE, '2019-05-03', '2021-05-03')

stock_file_name = '005930.KS2.csv'
encoding = 'euc-kr'  # 문자 인코딩
names = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
raw_dataframe = pd.read_csv(stock_file_name, names=names, encoding=encoding)  # 판다스이용 csv파일 로딩
raw_dataframe.info()  # 데이터 정보 출력
del raw_dataframe['Date']  # 위 줄과 같은 효과
stock_info = raw_dataframe.values[1:].astype(np.float)  # 금액&거래량 문자열을 부동소수점형으로 변환한다
print("stock_info.shape: ", stock_info.shape)
print("stock_info[0]: ", stock_info[0])
ori_price = stock_info[:, :-1]
# print(ori_price)
# print(ori_price[:,3])
ori_close_price = ori_price[:, 3]
ori_close_final_day_price = ori_close_price[-1]

# stock['Year'] = stock.index.year
# stock['Month'] = stock.index.month
# stock['Day'] = stock.index.day


scaler = MinMaxScaler()
scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
scaled = scaler.fit_transform(stock[scale_cols])
df = pd.DataFrame(scaled, columns=scale_cols)

x_train, x_test, y_train, y_test = train_test_split(df.drop('Close', 1), df['Close'], test_size=0.2, random_state=0,
                                                    shuffle=False)


def windowed_dataset(series, window_size, batch_size, shuffle):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    return ds.batch(batch_size).prefetch(1)


WINDOW_SIZE = 10
BATCH_SIZE = 32
# train_data 는 학습용 데이터셋, test_data 는 검증용 데이터셋 입니다.
train_data = windowed_dataset(y_train, WINDOW_SIZE, BATCH_SIZE, True)
test_data = windowed_dataset(y_test, WINDOW_SIZE, BATCH_SIZE, False)

# 아래의 코드로 데이터셋의 구성을 확인해 볼 수 있습니다.
# X: (batch_size, window_size, feature)
# Y: (batch_size, feature)
for data in train_data.take(1):
    print(f'데이터셋(X) 구성(batch_size, window_size, feature갯수): {data[0].shape}')
    print(f'데이터셋(Y) 구성(batch_size, window_size, feature갯수): {data[1].shape}')

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
loss = Huber()
optimizer = Adam(0.0005)
model.compile(loss=Huber(), optimizer=optimizer, metrics=['mse'])

# early_stopping 은 10번 epoch 동안 val_loss 개선이 없다면 학습을 멈춥니다.
# early_stopping = EarlyStopping(monitor='val_loss', patience=10)
# val_loss 기준 체크포인터도 생성합니다.
filename = os.path.join('tmp', 'ckeckpointer.ckpt')
checkpoint = ModelCheckpoint(filename,
                             save_weights_only=True,
                             save_best_only=True,
                             monitor='val_loss',
                             verbose=1)

history = model.fit(train_data,
                    validation_data=(test_data),
                    epochs=50,
                    callbacks=[checkpoint])

model.load_weights(filename)
pred = model.predict(test_data)
print(pred.shape)
print(pred[0])
print(pred[1])


# 역정규화 : 정규화된 값을 원래의 값으로 되돌림
def reverse_min_max_scaling(org_x, x):  # 종가 예측값
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()


# plt.figure(figsize=(12, 9))
# plt.plot(np.asarray(y_test)[5:], label='actual')
# plt.plot(pred, label='prediction')
# plt.legend()
# plt.show()

print(ori_price)
pred = reverse_min_max_scaling(ori_close_price, pred)  # 역정규화
print(pred)
print(pred[0])
