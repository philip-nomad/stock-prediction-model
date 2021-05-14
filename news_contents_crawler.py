import datetime
import os
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

PATH = "./"
os.chdir(PATH)
NEWS_DIR = 'news'

if not os.path.exists(NEWS_DIR):
    os.makedirs(NEWS_DIR)


def mkdir(company_code):
    if not os.path.exists(f"./{NEWS_DIR}/{company_code}/"):
        os.makedirs(f"./{NEWS_DIR}/{company_code}")


def start(company_code, crawling_target_date):
    print(f"company_code: {company_code} 뉴스기사 크롤링 시작")
    mkdir(company_code)

    unique_news_titles = set()
    page = 1
    while True:
        url = 'https://finance.naver.com/item/news_news.nhn?code=' + str(company_code) + '&page=' + str(page)

        source_code = requests.get(url).text
        html = BeautifulSoup(source_code, "lxml")

        processing_date = datetime.date.today()  # 오늘 날짜부터 시작

        dates = [datetime.datetime.strptime(date.get_text(), ' %Y.%m.%d %H:%M').date() for date in html.select('.date')]
        titles = [re.sub('\n', '', str(title.get_text())) for title in html.select('.title')]
        links = ['https://finance.naver.com' + link.find('a')['href'] for link in html.select('.title')]

        flag = True

        result_date = []
        result_title = []
        result_contents = []

        for row in list(zip(dates, titles, links)):
            date = row[0]
            title = row[1]
            link = row[2]

            if title in unique_news_titles:
                continue

            unique_news_titles.add(title)

            source_code = requests.get(link).text
            html = BeautifulSoup(source_code, "lxml")
            contents = str(html.select("div#news_read"))
            contents.find("<span")
            a = contents.find("<a")
            contents = remove_filename(contents[0:a])

            if (processing_date - date).days != 0:  # row 단위로 뉴스기사를 읽어오다가 날짜가 달라진 경우
                result = {"날짜": result_date, "기사제목": result_title, "본문내용": result_contents}
                df_result = pd.DataFrame(result)
                df_result.to_csv(
                    "./date_news/" + company_code + "/" + company_code + "_" + str(processing_date)[:10] + '.csv',
                    mode='w',
                    encoding='utf-8-sig'
                )
                processing_date = date
                result_date.clear()
                result_title.clear()
                result_contents.clear()

            if (processing_date - crawling_target_date).days < 0:  # 현재 읽어오려는 뉴스기사의 날짜가 원하는 날짜보다 더 과거의 날짜인 경우
                flag = False
                break

            result_date.append(date)
            result_title.append(title)
            result_contents.append(contents)

        if not flag:
            break

        print(f"company_code: {company_code}, processing_date: {processing_date}, 크롤링 끝난 페이지: {page}")
        page += 1


# html 태그 제거하는 코드
def remove_filename(content):
    cleaner = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')  # <tag>, &nbsp 등등 제거
    content = re.sub(cleaner, ' ', content)
    while content[-1] == '.':
        content = content[:-1]  # 끝에 . 제거 ex) test... -> test
        non_directory_letter = ['/', ':', '*', '?', '<', '>', '|']  # 경로 금지 문자열 제거
        for str_ in non_directory_letter:
            if str_ in content:
                content = content.replace(str_, "")

    # 영문제거
    cleaned_text = re.sub('[a-zA-Z]', ' ', content)
    # 이메일 제거
    cleaned_text = re.sub('[-=+,#/?:^$.@*\"※~&%ㆍ!』‘|()[]<>`\'…》]', ' ', cleaned_text)
    # 한글 아닌거 제거
    cleaned_text = re.sub(r'[^ㄱ-ㅣ가-힣]', ' ', cleaned_text)
    # 공백 2칸 이상을 1칸으로 수정
    cleaned_text = re.sub(' +', ' ', cleaned_text)
    # 문장 첫과 끝에 공백 제거
    cleaned_text = cleaned_text.strip()

    return cleaned_text
