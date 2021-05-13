import kosac_preprocessor
import news_contents_crawler
import news_contents_sentimental_analysis
import per_crawler
import prediction

company_code_list = ['005930','035720','035420','000660']

if __name__ == '__main__':
    for company_code in company_code_list:
        # 1. PER 정보 크롤링
        #per_crawler.start(company_code)
        # 2. 뉴스기사 크롤링
        news_contents_crawler.start(company_code)
        # 3. kosac 을 이용해 뉴스기사 데이터 전처리
        #kosac_preprocessor.start(company_code)
        # 4. 뉴스기사 감성분석
        #news_contents_sentimental_analysis.analyze(company_code)
        # 5. 주가예측
        #predicted_value = prediction.predict(company_code_list)

        #print(f"company_code: {company_code}")
        #print(f"value_predicted: {predicted_value}%")
