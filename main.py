import csv

import LSTM.LSTM_samsung_20 as LSTM
import kosac_preprocessor
import news_contents_crawler
import news_contents_sentimental_analysis
import per_crawler

company_code_list = ['005930']


def prediction(lstm, company_code):
    emotion_predict = []
    per_predict = []
    with open('./rate/' + company_code + '.csv', 'r', -1, 'utf-8') as rates:
        next(rates)
        for rate in csv.reader(rates):
            emotion_predict.append(rate[1])

    with open('./per_data/' + company_code + '.csv', 'r', -1, 'utf-8') as pers:
        next(pers)
        for per in csv.reader(pers):
            per_predict.append(per[3])
            per_predict.append(per[4])

    per = (1 - float(per_predict[0]) / float(per_predict[1]))
    print(f"PER 을 이용한 예측 점수:{per}")
    print(f"감성 분석을 이용한 예측 점수: {emotion_predict}")
    print(f"LSTM 예측 변동률: {lstm}")
    return 0.3 * float(emotion_predict[-1]) + 0.6 * lstm + per * 0.1


if __name__ == '__main__':
    cost = LSTM.lstm_samsung()
    print(cost)
    news_contents_crawler.start(company_code_list)
    per_crawler.start(company_code_list)
    for company_code in company_code_list:
        kosac_preprocessor.start(company_code)
        news_contents_sentimental_analysis.text_processing(company_code)
    print(f"삼성전자 4월 30일 종가:{cost[1]}")
    print(f"삼성전자 5월 3일 주가 종가 예측 변동률: {prediction((cost[0] - cost[1]) / cost[1], company_code_list[0]) * 100}%")
