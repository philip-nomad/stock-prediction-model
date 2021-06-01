import csv
import datetime
import os

import lstm_calculator
import news_contents_sentimental_analysis

PATH = "./"
os.chdir(PATH)
DIR = 'prediction_score'
STOCK_DIR = 'stock'


def predict(company_code, predict_date, w1, w2, w3):
    predict_date -= datetime.timedelta(days=1)
    with open(f"./{lstm_calculator.DIR}/{company_code}/{company_code}_{predict_date}.csv", 'r', -1,
              'utf-8') as lines:
        next(lines)

        for line in csv.reader(lines):
            lstm_prediction = round(float(line[0]))
            previous_closing_price = round(float(line[1]))

    lstm_value = (lstm_prediction - previous_closing_price) / previous_closing_price  # (다음날 예측 종가 - 오늘 종가) / 오늘 종가
    # 감성분석 값 불러오기
    emotional_analysis_csv = news_contents_sentimental_analysis.calculate_two_weeks(company_code,
                                                                                    predict_date)

    # PER 값 불러오기
    company_per_csv = 0
    same_category_per_csv = 0
    try:
        with open('./per_data/' + company_code + '.csv', 'r', -1, 'utf-8') as lines:
            next(lines)

            for line in csv.reader(lines):
                company_per_csv = float(line[3])
                same_category_per_csv = float(line[4])
    except FileNotFoundError:
        company_per_csv = 0
        same_category_per_csv = 0

    # 자기 회사 PER 이랑 동일업종 PER 이 모두 양수인 경우에만 per_value 계산
    if company_per_csv > 0 and same_category_per_csv > 0:
        per_value_csv = 1 - company_per_csv / same_category_per_csv  # 1 - 자기 회사 PER / 동일 업종 PER
    else:
        per_value_csv = 0

    predicted_value = ((lstm_value * w1 + emotional_analysis_csv * w2 + (
            per_value_csv * w3)) / 10 + 1) * previous_closing_price
    predicted_value = round(predicted_value / 10) * 10

    predict_date += datetime.timedelta(days=1)
    print(f"{company_code} {predict_date.strftime('%Y-%m-%d')} 예측 종가: {predicted_value}")

    return predicted_value
