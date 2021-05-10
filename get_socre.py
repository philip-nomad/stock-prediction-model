import preprocessor_criteria
import news_crawler_criteria
import sentimental_analysis_criteria
import make_criteria
#삼성전자:005930
#카카오:035720
#네이버:035420
#하이닉스:000660
#셀트리온:068270
company_code = '000660'
days = 0
wherep = 1
news_crawler_criteria.start(company_code,days,wherep)
preprocessor_criteria.start(company_code)
sentimental_analysis_criteria.analyze(company_code)
make_criteria.get_criteria(company_code,days)
#company_critera에 각 날짜별로 점수와 기사 개수가 저장됨
#저장 순서 0,ratio,portion,기사개수,date