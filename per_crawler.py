import datetime
import json
import os

import pandas as pd
import requests
from bs4 import BeautifulSoup

import elasticsearch_client

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

    # calculated per 계산
    calculated_per = 0
    if float(per) > 0 and float(category_per) > 0:
        calculated_per = 1 - float(per) / float(category_per)  # 1 - 자기 회사 PER / 동일 업종 PER

    # 결과 json 으로 다운로드
    today = str(datetime.date.today())
    result = {
        "@timestamp": str(datetime.datetime.now()),
        "date": today,
        "company_name": company_name,
        "company_code": company_code,
        "per": per,
        "same_category_per": category_per,
        "calculated_per": calculated_per,
    }
    json_data = json.dumps(result, ensure_ascii=False)

    path = 'per_data/json/' + company_code + '.json'

    with open(path, "w", encoding='UTF-8-sig') as file:
        file.write(json_data)

    # elasticsearch 로 데이터 전송
    index = f"per-data-{today}"
    elasticsearch_client.store_record(index, json_data)
