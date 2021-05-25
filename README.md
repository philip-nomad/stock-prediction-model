# stock-prediction-model

## versions
- python: 3.7

## Changes

#### 2021.4.5

- 코드 입력받지 않고 main.py 에서 list 형식으로 입력받게 변경
- 감성 분석을 각 회사 코드마다 따로 수행해서 각 csv 파일로 만들도록 변경
- 감성분석시에 한 글자로 되있는 단어 '못','안','않' 제외 하고 나머지는 감성 분석 하지 않도록 변경

#### 2021.4.12

- 뉴스 크롤링 할 시에 당일 크롤링 하도록 코드 변경
- 형태소 분석 코드 추가
- 감성 분석 변경(불필요한 단어 제거 코드 부분 삭제)
- 모든 .csv 파일 각 디렉토리에 저장하도록 변경

#### 2021.5.3

- 뉴스 점수 내는 부분에서 ratio, 점수에 산출하는 단어 리스트 따로 csv로 저장
- 산출방식을 기존의 수식에서 (부정 총점수) / (긍정 총점수) 로 수정
- 기준은 0.6으로 선정(이상은 부정, 아래는 긍정)
- 20단계로 나누어서 가중치 산정

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
- 10페이지 크롤링 실시 단 현재 날짜와 같은 뉴스들만 수집하도록 코드 변경

### How(PreProcess)

```python
def preprocess(code):
```

- Hannanum 형태소 분석기를 사용하여서 명사만 뽑아내여서 다시 news 크롤링 형식과 유사하게 다시 csv 파일 생성

### How(Analysis)

```python
def text_processing(code)
```

- main.py 에서 코드를 받아서 이미 크롤링된 텍스트들을 기반으로 감성분석 진행
- PreProcess 로 형태소 추출된 단어들을 기반으로 2단계 진행
  1. 불용어 제거
  2. Polarity 의 매칭되는 단어를 통해 부정, 중립, 긍정으로 환산
- 다시 .csv 파일로 저장하는 과정을 거친다.

### Libraries

- pandas - 대량의 데이터 처리
- requests - HTTP 통신
- bs4(beautifulSoup) - 크롤링
- konlpy.tag - 형태소 분석

### SubFiles

- company_list 
  - 회사 이름 - 종목 코드
- 불용어.txt
- polarity.csv
- news, score, words, score_words, ratio 디렉토리