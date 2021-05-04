import per_crawler
import news_contents_crawler as crawler
import news_contents_sentiment as sentiment
import preprocess_kosac as preprocess

# 삼전, SK하이닉스, NAVER, 카카오, 현대차, LG, SK, KT, 넷마블, 셀트리온, LG 화학, LG생활건강, 기아, 삼성전기, 이마트
company_code_list = ['005930', '000660', '035420', '035720','005380','003550','034730','030200','251270','068270','051910','051900','000270','009150','139480','008770']

if __name__ == '__main__':
    # start PER crawling
    per_crawler.start()

    # start NEWS crawling
    crawler.start(company_code_list)
    for company_code in company_code_list:
        preprocess.start(company_code)
        sentiment.text_processing(company_code)
