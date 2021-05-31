import datetime
import json
import operator

from elasticsearch import Elasticsearch

import prediction
from news_contents_sentimental_analysis import get_lstm_prediction_data, get_sentimental_score, get_score_word
import closing_calculation

HOST = 'stock-prediction-model.es.ap-northeast-2.aws.elastic-cloud.com'
PORT = '9243'
# TODO: api_key 환경변수로 등록하기
api_key = 'NTFObGNIa0JhOGU5ZnM2TDg2dmU6eEtvd0lHMXFSM1czZ284UUFydF9LZw=='
USERNAME = 'elastic'
PASSWORD = 'QO3LBFuS3GTOGNmATI2Lc9Vl'

"""
    1.1 company_score_word 
    {
        "company_code": "companyCode(string)"
        "company_name": "(string)"
        "date": "해당 날짜(string, ex: 2021-04-14)"
        "title" "(string)"
        "score_word": "샘플(stirng, split)"
        -> split 해서 하나씩 보내자!
    }

    1.2 company_score_word

    2. company_sentimental_score(company code 별)
    {
        "company_code: "company_code(string)"
        "company_name": "(string)"
        "negative_sum": Float Type
        "neutral_sum": Float Type
        "positive_sum": Float Type
        "date": "해당 날짜(string, ex: 2021-04-14)" 
    } 

    3.  per_data
    {
        "company_code: "company_code(string)"
        "company_name": "(string)"
        "date": "해당 날짜(string, ex: 2021-04-14)"
        "per": Float Type
        "same_category_per: Float Type
        "calculated_per": Float Type (0~1 사이 값)
    }

    4. company_sentimental_ratio 
    {
        "company_code: "company_code(string)"
        "company_name": "(string)"
        "date": "해당 날짜(string, ex: 2021-04-14)"
        "ratio": Float Type
        "prediction_percentage": Float Type
    }

    5. company_real_stock_data
    {
        "company_code: "company_code(string)"
        "company_name": "(string)"
        "date": "해당 날짜(string, ex: 2021-04-14)"
        "closing_price": Float Type
    }

    6. company_lstm
    {
        "company_code: "company_code(string)"
        "company_name": "(string)"
        "date": "해당 날짜(string, ex: 2021-04-14)"
        "lstm_prediction_price" : Float Type
        "lstm_prediction_ratio" : Float Type 
    }
    """


def store_record(index, json_data):
    elasticsearch = Elasticsearch([f"https://{USERNAME}:{PASSWORD}@{HOST}:{PORT}"])
    is_stored = True

    try:
        outcome = elasticsearch.index(index=index, doc_type='_doc', body=json_data)
        print(outcome)
    except Exception as ex:
        print('Error in indexing data')
        print(str(ex))
        is_stored = False
    finally:

        return is_stored


def post_data(company_code, company_name, start_date, end_date):
    post_json_word(company_code, company_name, start_date, end_date)
    post_score_json(company_code, company_name, start_date, end_date)
    post_json_prediction_accuracy(company_code, company_name, start_date, end_date)
    # 각각 가중치를 elasticsearch로 보내는 함수
    post_json_weight(company_code, company_name, start_date, end_date)


def post_json_word(company_code, company_name, start_date, end_date):
    while start_date <= end_date:
        words_list = get_score_word(company_code, start_date)
        word_dict = {}
        for word in words_list.split(" "):
            if len(word) >= 2:
                if word in word_dict:
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1

        sorted_word_tuple = sorted(word_dict.items(), key=operator.itemgetter(1), reverse=True)
        for idx, item in enumerate(sorted_word_tuple):
            if idx == 1:
                break
            else:
                result = {
                    "date": str(start_date),
                    "word": str(item[0]),
                    "count": int(item[1]),
                    "company_name": str(company_name),
                    "company_code": str(company_code)
                }

                print(result)
                json_data = json.dumps(result, ensure_ascii=False)
                # elasticsearch 로 데이터 전송
                index = f"word-count-{start_date}"

                store_record(index, json_data)

        start_date += datetime.timedelta(days=1)


def post_score_json(company_code, company_name, start_date, end_date):
    while start_date <= end_date:
        score_list = get_sentimental_score(company_code, start_date)

        result = {
            "date": str(start_date),
            "negative_sum": float(score_list[0]),
            "neutral_sum": float(score_list[1]),
            "positive_sum": float(score_list[2]),
            "company_name": str(company_name),
            "company_code": str(company_code),
        }
        json_data = json.dumps(result, ensure_ascii=False)

        # elasticsearch 로 데이터 전송
        index = f"news-score-{start_date}"
        store_record(index, json_data)

        start_date += datetime.timedelta(days=1)


def post_json_prediction_accuracy(company_code, company_name, start_date, end_date):
    while start_date <= end_date:
        w1, w2, w3 = prediction.start(company_code, start_date)

        # 바뀐 방식으로 예측 종가를 계산
        next_date = start_date + datetime.timedelta(days=1)
        predicted_value = closing_calculation.predict(company_code, next_date, w1, w2, w3)
        predicted_value = round(predicted_value / 10) * 10

        lstm_price, closing_price = get_lstm_prediction_data(company_code, start_date - datetime.timedelta(days=1))
        _, correct_closing_price = get_lstm_prediction_data(company_code, start_date)
        result = {
            "date": str(start_date),
            "lstm_closing_price": float(lstm_price),
            "correct_closing_price": float(correct_closing_price),
            "predicted_closing_price": float(predicted_value),
            "company_name": str(company_name),
            "company_code": str(company_code),
        }
        # print(result)
        # elasticsearch 로 데이터 전송
        json_data = json.dumps(result, ensure_ascii=False)
        index = f"prediction-accuracy-{start_date}"
        store_record(index, json_data)

        start_date += datetime.timedelta(days=1)


def post_json_weight(company_code, company_name, start_date, end_date):
    while start_date <= end_date:
        w1, w2, w3 = prediction.start(company_code, start_date)

        result = {
            "date": str(start_date),
            "lstm_weight": float(w1),
            "emotional_weight": float(w2),
            "per_weight": float(w3),
            "company_name": str(company_name),
            "company_code": str(company_code),
        }
        # print(result)
        # elasticsearch 로 데이터 전송
        json_data = json.dumps(result, ensure_ascii=False)
        index = f"prediction-weight-{start_date}"
        store_record(index, json_data)

        start_date += datetime.timedelta(days=1)
