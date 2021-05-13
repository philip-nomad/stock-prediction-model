import csv
import datetime
import os

import pandas as pd
from konlpy.tag import Hannanum

hannanum = Hannanum()


def start(company_code, target_date):
    start_date = datetime.date.today()
    while True:
        if (start_date - target_date).days < 0:  # 원하는 날짜보다 더 과거의 날짜인 경우
            break

        date_results = []
        title_results = []
        contents_results = []

        input_titles = []
        input_contexts = []

        try:
            with open("./date_news/" + company_code + "/" + company_code + "_" + str(start_date)[:10] + '.csv', 'r', -1,
                      'utf-8') as news:
                next(news)

                for line in csv.reader(news):
                    date_results.append(line[1])
                    title_results.append(line[2])
                    contents_results.append(line[3])
        except FileNotFoundError:
            start_date = start_date - datetime.timedelta(days=1)
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

            if not os.path.exists("./date_news_words/" + company_code):
                os.makedirs("./date_news_words/" + company_code)

            f = open("./date_news_words/" + company_code + "/" + company_code + "_" + str(start_date)[:10] + '.csv',
                     "w+")
            f.close()
            columns = ['time', 'title', 'context']
            df = pd.DataFrame(columns=columns)
            df["time"] = date_results
            df["title"] = input_titles
            df["context"] = input_contexts
            df.to_csv(
                "./date_news_words/" + company_code + "/" + company_code + "_" + str(start_date)[:10] + '.csv',
                index=False
            )
            pd.read_csv("./date_news_words/" + company_code + "/" + company_code + "_" + str(start_date)[:10] + '.csv')

            start_date = start_date - datetime.timedelta(days=1)
