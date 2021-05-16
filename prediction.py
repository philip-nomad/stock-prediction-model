import csv

import lstm_calculator
import news_contents_sentimental_analysis

WEIGHT_FOR_LSTM_VALUE = 0.6  # 가중치 a: LSTM 가중치
WEIGHT_FOR_EMOTIONAL_ANALYSIS_VALUE = 0.3  # 가중치 b: 감성분석 점수 가중치
WEIGHT_FOR_PER_VALUE = 0.1  # 가중치 c: PER 점수 가중치


def start(company_code, learning_date):
    print(f"company_code: {company_code} 다음 날 종가 예측 시작")

    # LSTM 값 불러오기
    with open(f"./{lstm_calculator.DIR}/{company_code}_{learning_date}.csv", 'r', -1, 'utf-8') as lines:
        next(lines)

        for line in csv.reader(lines):
            lstm_prediction = int(line[0])
            closing_price = int(line[1])

    lstm_value = (lstm_prediction - closing_price) / closing_price  # (다음날 예측 종가 - 오늘 종가) / 오늘 종가
    print(f"LSTM 예측 변동률: {lstm_value}")

    # 감성분석 값 불러오기
    emotional_analysis_value = news_contents_sentimental_analysis.calculate_two_weeks(company_code, learning_date)
    print(f"감성 분석을 이용한 예측 점수: {emotional_analysis_value}")

    # PER 값 불러오기
    company_per = 0
    same_category_per = 0
    with open('./per_data/' + company_code + '.csv', 'r', -1, 'utf-8') as lines:
        next(lines)

        for line in csv.reader(lines):
            company_per = float(line[3])
            same_category_per = float(line[4])

    # 자기 회사 PER 이랑 동일업종 PER 이 모두 양수인 경우에만 per_value 계산
    if company_per > 0 and same_category_per > 0:
        per_value = 1 - company_per / same_category_per  # 1 - 자기 회사 PER / 동일 업종 PER
    else:
        per_value = 0

    print(f"PER 을 이용한 예측 점수:{per_value}")

    # 가중치 a, b, c 곱해서 예측점수 구하기
    predicted_value = 100 * (lstm_value * WEIGHT_FOR_LSTM_VALUE +
                             float(emotional_analysis_value) * WEIGHT_FOR_EMOTIONAL_ANALYSIS_VALUE +
                             per_value * WEIGHT_FOR_PER_VALUE)

    print(f"company_code: {company_code}")
    print(f"learning_date: {learning_date}")
    print(f"predicted_value: {predicted_value}%")

######### 모델 a,b,c
#lstm_data =
#emotional_data =
#per_data =