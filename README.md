# stock-prediction-model

## versions
- python: 3.8.2

## Changes

#### 2021.4.5

- 코드 입력받지 않고 main.py에서 list 형식으로 입력받게 변경
- 감성 분석을 각 회사 코드마다 따로 수행해서 각 csv파일로 만들도록 변경
- 감성분석시에 한 글자로 되있는 단어 '못','안','않' 제외 하고 나머지는 감성 분석 하지 않도록 변경

### How (crawling)

```python
def start():
```

- 크롤링 시작 (종목 이름이나 코드를 받음)
- convert_to_code로 연결

```python
def convert_to_code(company, maxpage):
```

- 이름과 종목코드로 분기점 나뉨

```python
def crawl(company_code, maxpage):
```

- 실질적인 크롤링 부분 title, content, date로 나뉘어서 (종목코드).csv 파일로 변환해서 저장

### How(Analysis)

```python
def text_processing(code)
```

- main.py에서 코드를 받아서 이미 크롤링된 텍스트들을 기반으로 감성분석 진행
- 한글만 추출 - 불용어 제거 - 감성분석 사전으로 감성 분석 으로 3단계 진행

### Libraries

- pandas - 대량의 데이터 처리
- requests - HTTP 통신
- bs4(beautifulSoup) - 크롤링

### SubFiles

- company_list 
  - 회사 이름 - 종목 코드
- 005930.csv
  - 예시 자료(삼성전자)
- 005930_score.csv
  - 예시 자료(삼성전자 각 기사 점수)