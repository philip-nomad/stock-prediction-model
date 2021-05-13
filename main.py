import datetime

import news_contents_sentimental_analysis

COMPANY_CODE_LIST = ['005930', '035720', '035420', '000660']
TARGET_DATE = datetime.date(2021, 2, 1)
TARGET_DATE_TWO_WEEKS = datetime.date(2021, 2, 14)

if __name__ == '__main__':
    for company_code in COMPANY_CODE_LIST:
        # 1. PER 정보 크롤링
        # per_crawler.start(company_code)
        # 2. 뉴스기사 크롤링
        # news_contents_crawler.start(company_code, TARGET_DATE)
        # 3. kosac 을 이용해 뉴스기사 데이터 전처리
        # kosac_preprocessor.start(company_code, TARGET_DATE)
        # 4. 뉴스기사 감성분석
        # news_contents_sentimental_analysis.analyze(company_code, TARGET_DATE)
        # 5. 주가예측
        # predicted_value = prediction.predict(company_code_list)
        print(news_contents_sentimental_analysis.analyze_two_weeks(company_code, TARGET_DATE_TWO_WEEKS))
        # print(f"company_code: {company_code}")
        # print(f"value_predicted: {predicted_value}%")
