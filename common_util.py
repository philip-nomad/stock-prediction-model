import json
import csv
from collections import OrderedDict

def csv_file_to_json(path, indexes, date=None):
    json_keys = []
    data = []

    with open(path,'r',-1,'utf-8') as lines:
        for index in indexes:
            json_keys.append(csv.reader(lines)[0][index])

        next(lines)
        tmp_data = []
        for index in indexes:
            for line in csv.reader(lines):
                tmp_data.append(line[index])
        data.append(tmp_data)


    # for key in json_keys:
    #     for  in data:


    json_file = None



    return json_file
