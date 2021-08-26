import re
import warnings

import requests
from bs4 import BeautifulSoup
import pandas as pd
from konlpy.tag import Okt

# 파일로 저장한 TEXT를 Konlpy를 이용하여 명사 단어를 추츨 한다.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

okt = Okt();
mdata = pd.read_excel('movies.xlsx');
print(mdata.head(10))
print(len(mdata))

mdata = mdata[mdata['text'].notnull()]
mdata['text'] = \
    mdata['text'].apply(lambda x:re.sub(r'[^ ㄱ-ㅣ가-힣 ]+', " ", x));
text = mdata['text'];
score = mdata['score'];

# 학습 텍스트와 테스트 텍스트로 나눈다.
# TfidfVectorizer를 이용하여 수치화 한다.

x_train, x_test, y_train, y_test = train_test_split(
    text, score, test_size=0.2, random_state=777
);
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))

warnings.filterwarnings('ignore');

tfv = TfidfVectorizer(tokenizer=okt.morphs, min_df=3, max_df=0.9);
print('Row Data ..')
print(x_train);
print(y_train);
print('TfidfVectorizer Data ..')
tfv_x_train = tfv.fit_transform(x_train);
print(tfv_x_train);

# 머신러닝을 이용하여 학습 한다.
# hyper parameter 가중치 1~22 의 값을 넣어줌
params = {'C':[5,6,9,11,15,18,20,22]};
lr = LogisticRegression();
grid_cv = GridSearchCV(lr, param_grid=params,cv=3,
                       scoring='accuracy', verbose=0);
print('Train ...')
grid_cv.fit(tfv_x_train,y_train);
print('End Train ...')
print(grid_cv.best_params_, grid_cv.best_score_)
# TEST 진행
tfv_x_test = tfv.transform(x_test);
test_pr = grid_cv.best_estimator_.predict(tfv_x_test);
print('정확도:',accuracy_score(y_test,test_pr));

# 실제 입력한 문장을 기반으로 예측
# 2
input_text = '배우들의 연기랑 화려한 액션신은 끝내준다';
# 1
#input_text = '마지막 닛산 스카이 라인이 나오는 장면 만으로도 이 영화는 가치가 있다';
# 0
#input_text = '스토리 진짜 역대급 쓰레기';
input_text = re.compile(r'[ ㄱ-ㅣ가-힣]+').findall(input_text)
input_text = [' '.join(input_text)];

tfv_input_text = tfv.transform(input_text);
#print(tfv_input_text)
result = grid_cv.best_estimator_.predict(tfv_input_text);
print('result:',result)
if result == 0:
    print('예측결과:-->> 부정');
elif result == 1:
    print('예측결과:-->> 긍정');
elif result == 2:
    print('예측결과:-->> 긍긍정');
