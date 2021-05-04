import csv
import per_crawler
import LSTM.LSTM_samsung_20 as LSTM
import news_contents_crawler as crawler
import news_contents_sentiment as sentiment
import preprocess_kosac as preprocess

# 삼전, SK하이닉스, NAVER, 카카오, 현대차, LG, SK, KT, 넷마블, 셀트리온, LG 화학, LG생활건강, 기아, 삼성전기, 이마트
#, '000660', '035420', '035720','005380','003550','034730','030200','251270','068270','051910','051900','000270','009150','139480','008770'
company_code_list = ['005930']

def prediction(lstm,company_code):
    emotion_predict = []
    per_predict = []
    per = 0
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
    print(f"PER을 이용한 예측 점수:{per}")
    print(f"감성 분석을 이용한 예측 점수: {emotion_predict}")
    print(f"LSTM 예측 변동률: {lstm}")
    return 0.3 * float(emotion_predict[-1]) + 0.6 * lstm + per * 0.1

#(1 - 삼성전자 PER / 동일업종 PER) * 0.1(=PER 가중치)



if __name__ == '__main__':
    # start PER crawling

    cost = LSTM.lstm_samsung()
    print(cost)
    crawler.start(company_code_list)
    per_crawler.start(company_code_list)
    for company_code in company_code_list:
        preprocess.start(company_code)
        sentiment.text_processing(company_code)
    print(f"삼성전자 4월 30일 종가:{cost[1]}")
    print(f"삼성전자 5월 3일 주가 종가 예측 변동률: {prediction((cost[0]-cost[1])/cost[1],company_code_list[0])*100}%")