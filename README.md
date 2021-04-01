# stock-prediction-model

## versions
- python: 3.8.2

### How (crawling)

```python
def start_crolling():
```

- 크롤링 시작 (종목 이름이나 코드를 받음)
- convert_to_code로 연결

```python
def convert_to_code(company, maxpage):
```

- 이름과 종목코드로 분기점 나뉨

```python
def crawler(company_code, maxpage):
```

- 실질적인 크롤링 부분 title, content, date로 나뉘어서 (종목코드).csv 파일로 변환해서 저장

### Libraries

- pandas - 대량의 데이터 처리
- requests - HTTP 통신
- bs4(beautifulSoup) - 크롤링

### SubFiles

- company_list 
  - 회사 이름 - 종목 코드
- 005930.csv
  - 예시 자료(삼성전자)