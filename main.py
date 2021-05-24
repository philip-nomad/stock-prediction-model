import datetime

import elasticsearch_client
import kosac_preprocessor
import lstm_calculator
import news_contents_crawler
import news_contents_sentimental_analysis
import per_crawler
import prediction

# COMPANIES = [('005930', '삼성전자'), ('035720', '카카오'), ('035420', '네이버'), ('000660', 'SK하이닉스')]
COMPANIES = [('005930', '삼성전자')]
START_DATE = datetime.date(2021, 4, 30)
END_DATE = datetime.date(2021, 5, 7)

# 11pm 에 돌릴 함수
if __name__ == '__main__':
    for company in COMPANIES:
        # 1. PER 정보 크롤링
        per_crawler.start(company[0])

        # 2. 뉴스기사 크롤링
        news_contents_crawler.start(company[0], START_DATE, END_DATE)

        # 3. kosac 을 이용해 뉴스기사 데이터 전처리
        kosac_preprocessor.start(company[0], START_DATE, END_DATE)

        # 4. 뉴스기사 감성분석
        news_contents_sentimental_analysis.start(company[0], START_DATE, END_DATE)

        learning_date = START_DATE
        while learning_date <= END_DATE:
            # 5. lstm 계산
            lstm_calculator.start(company[0], learning_date)

            # 6. 가중치 a, b, c 를 활용하여 주가예측
            predicted_value = prediction.start(company[0], learning_date)

            learning_date += datetime.timedelta(days=1)

        # 7. elasticsearch 로 데이터 전송
        elasticsearch_client.post_data(company[0], company[1], START_DATE, END_DATE)
