import csv
import datetime
import os

import pandas as pd
from konlpy.tag import Hannanum

import news_contents_crawler

hannanum = Hannanum()

NEWS_WORDS_DIR = 'news_words'


def mkdir(company_code):
    if not os.path.exists(f"./{NEWS_WORDS_DIR}/{company_code}"):
        os.makedirs(f"./{NEWS_WORDS_DIR}/{company_code}")


def start(company_code, start_date, end_date):
    print(f"company_code: {company_code} 뉴스기사 전처리 시작")
    mkdir(company_code)

    while start_date <= end_date:
        date_results = []
        title_results = []
        contents_results = []

        input_titles = []
        input_contexts = []

        try:
            with open(f"./{news_contents_crawler.NEWS_DIR}/{company_code}/{company_code}_{str(start_date)[:10]}.csv",
                      'r',
                      -1,
                      'utf-8') as news:
                next(news)

                for line in csv.reader(news):
                    date_results.append(line[1])
                    title_results.append(line[2])
                    contents_results.append(line[3])
        except FileNotFoundError:
            print('FileNotFoundError: 해당 날짜에 뉴스기사가 존재하지 않습니다.')
        else:
            for title in title_results:
                text = hannanum.nouns(title)
                input_title = ""
                for t in text:
                    input_title += t + " "
                input_title = input_title.strip()
                input_titles.append(input_title)

            for content in contents_results:
                text = hannanum.nouns(content)
                input_content = ""
                for t in text:
                    input_content += t + " "
                input_content = input_content.strip()
                input_contexts.append(input_content)

            f = open(f"./{NEWS_WORDS_DIR}/{company_code}/{company_code}_{str(start_date)[:10]}.csv",
                     "w+")
            f.close()
            columns = ['time', 'title', 'context']
            df = pd.DataFrame(columns=columns)
            df["time"] = date_results
            df["title"] = input_titles
            df["context"] = input_contexts
            df.to_csv(
                f"./{NEWS_WORDS_DIR}/{company_code}/{company_code}_{str(start_date)[:10]}.csv",
                index=False
            )

        finally:
            start_date += datetime.timedelta(days=1)
