import csv

import LSTM.LSTM_samsung_20 as LSTM

WEIGHT_FOR_LSTM_VALUE = 0.6
WEIGHT_FOR_EMOTIONAL_ANALYSIS_VALUE = 0.3
WEIGHT_FOR_PER_VALUE = 0.1


def predict(company_code):
    lstm_price = LSTM.lstm_samsung()
    lstm_value = (lstm_price[0] - lstm_price[1]) / lstm_price[1]
    print(f"LSTM 예측 변동률: {lstm_value}")

    emotional_analysis_values = []
    with open('./rate/' + company_code + '.csv', 'r', -1, 'utf-8') as rates:
        next(rates)

        for rate in csv.reader(rates):
            emotional_analysis_values.append(rate[1])
    print(f"감성 분석을 이용한 예측 점수: {emotional_analysis_values}")

    company_per = 0
    same_category_per = 0
    with open('./per_data/' + company_code + '.csv', 'r', -1, 'utf-8') as lines:
        next(lines)

        for line in csv.reader(lines):
            company_per = float(line[3])
            same_category_per = float(line[4])

    if company_per >= 0 and same_category_per >= 0:
        per_value = 1 - company_per / same_category_per  # 1 - 자기 회사 PER / 동일 업종 PER
    else:
        per_value = 0

    print(f"PER 을 이용한 예측 점수:{per_value}")

    predicted_value = 100 * (lstm_value * WEIGHT_FOR_LSTM_VALUE +
                             float(emotional_analysis_values[-1]) * WEIGHT_FOR_EMOTIONAL_ANALYSIS_VALUE +
                             per_value * WEIGHT_FOR_PER_VALUE)

    return predicted_value
