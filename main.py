# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import news_contents_crawler as crawler
import news_contents_sentiment as sentiment
import preprocess_kosac as preprocess

# 삼전, SK하이닉스, NAVER, 카카오, 현대차, LG, SK, KT, 넷마블, 셀트리온, LG 화학, LG생활건강, 기아, 삼성전기, 이마트
company_code_list = ['005930', '000660', '035420', '035720','005380','003550','034730','030200','251270','068270','051910','051900','000270','009150','139480','008770']

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # start crawling
    #crawler.start(company_code_list)
    for company_code in company_code_list:
        preprocess.start(company_code)
        sentiment.text_processing(company_code)
