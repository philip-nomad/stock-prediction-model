# Hannanum
from konlpy.tag import Hannanum
import pandas as pd
import csv

hannanum = Hannanum()

def preprocess(code):
    time_result = []
    title_result = []
    context_result = []

    input_titles = []
    input_contexts = []

    with open("./news/"+code+'.csv', 'r', -1, 'utf-8') as news:
        next(news)

        for line in csv.reader(news):
            time_result.append(line[1])
            title_result.append(line[2])
            context_result.append(line[3])

    for title in title_result:
        text = hannanum.nouns(title)
        str = ""
        for t in text:
            str += t + " "
        input_titles.append(str)

    for context in context_result:
        text = hannanum.nouns(context)
        str = ""
        for t in text:
            str += t + " "
        input_contexts.append(str)
    f = open("./words/" + code + '.csv', "w+")
    f.close()
    columns = ['time', 'title', 'context']
    df = pd.DataFrame(columns=columns)
    df["time"] = time_result
    df["title"] = input_titles
    df["context"] = input_contexts
    df.to_csv("./words/"+code+'.csv', index=False)
    ds = pd.read_csv("./words/"+code+'.csv')