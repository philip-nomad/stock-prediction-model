# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import news_contents_crawler as crawler
import news_contents_sentiment as sentiment
import preprocess_kosac as preprocess

# 삼전, SK하이닉스, NAVER, 카카오
code_lists = ['005930','000660','035420','035720']

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # start crawling
    crawler.start(code_lists)
    for code in code_lists:
        preprocess.preprocess(code)
    for code in code_lists:
        sentiment.text_processing(code)
