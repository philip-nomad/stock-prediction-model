# Hannanum
import csv

import pandas as pd
from konlpy.tag import Hannanum

hannanum = Hannanum()


def start(company_code):
    date_results = []
    title_results = []
    content_results = []

    input_titles = []
    input_contexts = []

    with open("./news/" + company_code + '.csv', 'r', -1, 'utf-8') as news:
        next(news)

        for line in csv.reader(news):
            date_results.append(line[1])
            title_results.append(line[2])
            content_results.append(line[3])

    for title in title_results:
        text = hannanum.nouns(title)
        input_title = ""
        for t in text:
            input_title += t + " "
        input_title = input_title.strip()
        input_titles.append(input_title)

    for content in content_results:
        text = hannanum.nouns(content)
        input_content = ""
        for t in text:
            input_content += t + " "
        input_content = input_content.strip()
        input_contexts.append(input_content)

    f = open("./words/" + company_code + '.csv', "w+")
    f.close()
    columns = ['time', 'title', 'context']
    df = pd.DataFrame(columns=columns)
    df["date"] = date_results
    df["title"] = input_titles
    df["context"] = input_contexts
    df.to_csv("./words/" + company_code + '.csv', index=False)
    pd.read_csv("./words/" + company_code + '.csv')
