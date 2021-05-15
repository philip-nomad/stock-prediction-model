import datetime

import kosac_preprocessor
import lstm_calculator
import news_contents_crawler
import news_contents_sentimental_analysis
import per_crawler
import prediction

COMPANY_CODE_LIST = ['035420', '000660']
#COMPANY_CODE_LIST = ['035720']
CRAWLING_TARGET_DATE = datetime.date(2021, 5, 14)  # 어느 날까지 크롤링을 할 것인가
LEARNING_DATE = datetime.date(2021, 2, 14)  # 어느 날까지 학습하여 그 다음 날 주가를 예측할 것인가
# 11pm 에 돌릴 함수
if __name__ == '__main__':
    for company_code in COMPANY_CODE_LIST:
        # 1. PER 정보 크롤링
        #per_crawler.start(company_code)
        # 2. 뉴스기사 크롤링
        #news_contents_crawler.start(company_code, CRAWLING_TARGET_DATE)
        # 3. kosac 을 이용해 뉴스기사 데이터 전처리
        #kosac_preprocessor.start(company_code, CRAWLING_TARGET_DATE)
        # 4. 뉴스기사 감성분석
        #news_contents_sentimental_analysis.start(company_code, CRAWLING_TARGET_DATE)
        # 5. lstm 계산
        lstm_calculator.start(company_code, LEARNING_DATE)
        # 6. 가중치 a, b, c 를 활용하여 주가예측
        #predicted_value = prediction.start(company_code, LEARNING_DATE)

