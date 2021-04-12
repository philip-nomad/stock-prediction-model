from bs4 import BeautifulSoup
import requests
import re
import os
import pandas as pd

PATH = "./"
os.chdir(PATH)
if not os.path.exists('per_data'):
    os.makedirs('per_data')


def crawler(company_code):
    url = 'https://finance.naver.com/item/main.nhn?code=' + str(company_code)
    source_code = requests.get(url).text
    html = BeautifulSoup(source_code, "lxml")
    company_name = html.find(attrs={'class': 'wrap_company'}).find('h2').text

    # 해당 종목 PER
    per_em_tag = html.find(id='_per')
    per = per_em_tag.text

    # 동종 업계 PER
    div = html.find(id='tab_con1')
    table = div.find(attrs={'summary': '동일업종 PER 정보'})
    category_per = table.find('em').text

    # 결과 다운로드
    result = {"company_name": [company_name], "company_code": [company_code], "PER": [per], "동일업종 PER": [category_per]}
    df_result = pd.DataFrame(result)
    df_result.to_csv('per_data/' + company_code + '.csv', mode='w', encoding='utf-8-sig')


# 회사명을 종목코드로 변환
def convert_to_code(company):
    data = pd.read_csv('company_list.txt', dtype=str, sep='\t')  # 종목코드 추출
    company_names = data['회사명'].tolist()
    company_codes = data['종목코드'].tolist()

    dict_result = dict(zip(company_names, company_codes))  # 딕셔너리 형태로 회사이름과 종목코드 묶기

    pattern = '[a-zA-Z가-힣]+'

    # Input 에 이름으로 넣었을 때
    if bool(re.match(pattern, company)):
        company_code = dict_result.get(str(company))

    # Input 에 종목코드로 넣었을 때
    else:
        company_code = str(company)

    return company_code


def start():
    input("=" * 50 + "\n" + "실시간 뉴스기사 다운받기." + "\n" + "시작하시려면 Enter 를 눌러주세요." + "\n" + "=" * 50)
    company_name_or_code = input("종목 이름이나 코드 입력: ")
    company_code = convert_to_code(company_name_or_code)
    crawler(company_code)
