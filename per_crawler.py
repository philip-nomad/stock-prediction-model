import os

import pandas as pd
import requests
from bs4 import BeautifulSoup

PATH = "./"
os.chdir(PATH)
if not os.path.exists('per_data'):
    os.makedirs('per_data')
if not os.path.exists('per_data/json'):
    os.makedirs('per_data/json')
if not os.path.exists('per_data/csv'):
    os.makedirs('per_data/csv')


def start(company_code):
    print(f"company_code: {company_code} PER 정보 크롤링 시작")

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

    # 결과 csv 로 다운로드
    result = {
        "company_name": [company_name],
        "company_code": [company_code],
        "per": [per],
        "same_category_per": [category_per]
    }
    df_result = pd.DataFrame(result)
    path = 'per_data/csv/' + company_code + '.csv'
    df_result.to_csv(path, mode='w', encoding='utf-8-sig', index=False)
