import os
import pandas as pd

from LSTM import LSTM_score

company_code_list = ['005930']

start_date = '2020-02-01'
end_date = '2021-02-01'
predict_date = '2021-02-02'
print('start')

PATH = "./"
os.chdir(PATH)
if not os.path.exists('company_lstm_score'):
    os.makedirs('company_lstm_score')

for company_code in company_code_list:
    print(company_code)
    lstm_score = LSTM_score.get_lstm_score(company_code, start_date, end_date)  # 폐,폐
    lstm_score=lstm_score.astype(int)
    score_columns = ['Date','Stock']
    score_df = pd.DataFrame(columns=score_columns)
    score_df['Date'] = [predict_date]
    score_df['Stock'] = [lstm_score.item(0)]
    score_df.to_csv("./company_lstm_score/" + company_code + '_score.csv', mode='a', index=False)
