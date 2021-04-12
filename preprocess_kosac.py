# Hannanum
from konlpy.tag import Hannanum
hannanum = Hannanum()

tf = open("test1.txt",'r',encoding='utf-8')
testfile = tf.read()

hannanum.analyze #구 Phrase 분석
hannanum.morphs # 형태소 분석
hannanum.nouns # 명사 분석
hannanum.pos # 형태소 분석 태깅
print("hannanum analyze")
a = hannanum.analyze(testfile)
print(a)
print("hannanum pos")
print(hannanum.pos(testfile))
#예시

#Kkma
from konlpy.tag import Kkma
kkma = Kkma()

kkma.morphs # 형태소 분석
kkma.nouns # 명사 분석
kkma.pos # 형태소 분석 태깅
kkma.sentences #문장 분석

print("kkma pos")
print(kkma.pos(testfile))
#예시

