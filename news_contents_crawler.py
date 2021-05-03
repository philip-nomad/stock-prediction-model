import os
import re
from datetime import datetime
from datetime import date

import pandas as pd
import requests
from bs4 import BeautifulSoup

PATH = "./"
os.chdir(PATH)
if not os.path.exists('news'):
    os.makedirs('news')

if not os.path.exists('score'):
    os.makedirs('score')

if not os.path.exists('words'):
    os.makedirs('words')


def crawl(company_code):
    page = 1
    title_result = []
    content_result = []
    date_res = []
    while True:
        is_able_to_crawl = False
        url = 'https://finance.naver.com/item/news_news.nhn?code=' + str(company_code) + '&page=' + str(page)
        source_code = requests.get(url).text
        html = BeautifulSoup(source_code, "lxml")

        is_able = []
        current_date = datetime.now().date()

        dates = [date.get_text() for date in html.select('.date')]
        for date in dates:
            date_compare = datetime.strptime(date, ' %Y.%m.%d %H:%M').date()
            delta = current_date - date_compare
            if delta.days <= 4:
                is_able_to_crawl = True
                is_able.append(True)
                date_res.append(date_compare)
            else:
                is_able.append(False)

        if not is_able_to_crawl:
            break

        titles = html.select('.title')
        for idx, title in enumerate(titles):
            if not is_able[idx]:
                continue
            title = title.get_text()
            title = re.sub('\n', '', title)
            title_result.append(title)

        # 뉴스 링크
        link_result = []
        links = html.select('.title')

        for link in links:
            add = 'https://finance.naver.com' + link.find('a')['href']
            link_result.append(add)

        for idx, link in enumerate(link_result):
            if not is_able[idx]:
                continue
            url = link
            source_code = requests.get(url).text
            html = BeautifulSoup(source_code, "lxml")
            contents = html.select("div#news_read")
            text = str(contents)
            text.find("<span")
            a = text.find("<a")
            text = remove_filename(text[0:a])
            content_result.append(text)

        page += 1

    result = {"날짜": date_res, "기사제목": title_result, "본문내용": content_result}
    df_result = pd.DataFrame(result)
    print(f"{company_code} 기사를 다운 받고 있습니다------")
    df_result.to_csv("./news/" + company_code + '.csv', mode='w', encoding='utf-8-sig')


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
    cleaned_text = re.sub('[-=+,#/?:^$.@*\"※~&%ㆍ!』‘|()\[\]<>`\'…》]', ' ', cleaned_text)
    # 한글 아닌거 제거
    cleaned_text = re.sub(r'[^ㄱ-ㅣ가-힣]', ' ', cleaned_text)
    # 공백 2칸 이상을 1칸으로 수정
    cleaned_text = re.sub(' +', ' ', cleaned_text)
    # 문장 첫과 끝에 공백 제거
    cleaned_text = cleaned_text.strip()

    return cleaned_text


def start(company_code_list):
    for company_code in company_code_list:
        crawl(company_code)
