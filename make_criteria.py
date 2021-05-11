from datetime import datetime, timedelta

import pandas as pd


def get_criteria(company_code, days):
    df = pd.read_csv("./criteria_rate" + "\\" + company_code + '.csv')
    get_score = df.iloc[-1]
    df_news_cnt = pd.read_csv("./criteria_news" + "\\" + company_code + '.csv')
    get_score['기사개수'] = df_news_cnt.shape[0]
    get_score['날짜'] = datetime.today() - timedelta(days)
    get_score.to_csv("./company_criteria" + "\\" + company_code + '.csv', mode='a', index=False)
