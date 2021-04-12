from bs4 import BeautifulSoup
import requests
import re
import os
import pandas as pd
import datetime

PATH = "./"
os.chdir(PATH)


def crawl(company_code, max_page):
    page = 1
    title_result = []
    content_result = []
    date_res = []
    while page <= int(max_page):
        can_craw = False
        url = 'https://finance.naver.com/item/news_news.nhn?code=' + str(company_code) + '&page=' + str(page)
        source_code = requests.get(url).text
        html = BeautifulSoup(source_code, "lxml")

        is_able = []
        current = datetime.datetime.now()
        current = datetime.date(current.year,current.month,current.day)
        dates = html.select('.date')
        date_result = [date.get_text() for date in dates]

        for d in date_result:
            day = d[1:11].split('.')
            date_compare = datetime.datetime.strptime(day[0]+day[1]+day[2],"%Y%m%d")
            date_compare = datetime.date(date_compare.year,date_compare.month,date_compare.day)
            date_string = date_compare.strftime('%Y-%m-%d')
            if current == date_compare:
                can_craw = True
                is_able.append(True)
                date_res.append(date_string)
            else :
                is_able.append(False)

        if can_craw is False :
            break

        idx = 0
        titles = html.select('.title')
        for title in titles:
            if is_able[idx] is False:
                continue
            title = title.get_text()
            title = re.sub('\n', '', title)
            title_result.append(title)
            idx += 1

        # 뉴스 링크
        link_result = []
        links = html.select('.title')

        for link in links:
            add = 'https://finance.naver.com' + link.find('a')['href']
            link_result.append(add)

        # 뉴스 날짜

        # 변수들 합쳐서 해당 디렉토리에 csv 파일로 저장하기
        idx = 0
        for link in link_result:
            if is_able[idx] is False:
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
            idx += 1
        page += 1

    result = {"날짜": date_res, "기사제목": title_result, "본문내용": content_result}
    df_result = pd.DataFrame(result)
    print("다운 받고 있습니다------")
    df_result.to_csv("./news/"+company_code + '.csv', mode='w', encoding='utf-8-sig')
    # 종목 리스트 파일 열기


# html 태그 제거하는 코드
def remove_filename(string):
    cleaner = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')  # <tag>, &nbsp 등등 제거
    string = re.sub(cleaner, ' ', string)
    while string[-1] == '.':
        string = string[:-1]  # 끝에 . 제거 ex) test... -> test
        non_directory_letter = ['/', ':', '*', '?', '<', '>', '|']  # 경로 금지 문자열 제거
        for str_ in non_directory_letter:
            if str_ in string:
                string = string.replace(str_, "")
    cleaned_text = re.sub('[a-zA-Z]','',string)
    cleaned_text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', ' ', cleaned_text)
    cleaned_text = re.sub(r'[^ ㄱ-ㅣ가-힣A-Za-z]', ' ', cleaned_text)
    return cleaned_text


# 회사명을 종목코드로 변환
def convert_to_code(company, max_page):
    data = pd.read_csv('company_list.txt', dtype=str, sep='\t')  # 종목코드 추출
    company_name = data['회사명']
    keys = [i for i in company_name]  # 데이터프레임에서 리스트로 바꾸기

    company_code = data['종목코드']
    values = [j for j in company_code]

    dict_result = dict(zip(keys, values))  # 딕셔너리 형태로 회사이름과 종목코드 묶기

    pattern = '[a-zA-Z가-힣]+'

    if bool(re.match(pattern, company)):  # Input에 이름으로 넣었을 때
        company_code = dict_result.get(str(company))
        crawl(company_code, max_page)

    # Input 에 종목코드로 넣었을 때
    else:
        company_code = str(company)
        crawl(company_code, max_page)


def start(code_lists):

    for code in code_lists:
        company = code
        max_page = 10
        convert_to_code(company, max_page)
