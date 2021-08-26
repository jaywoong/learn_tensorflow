import requests
from bs4 import BeautifulSoup
import pandas as pd
from konlpy.tag import Okt

pages = 50;
#movie_code = [189150,191920];
movie_code = [189150, 191920, 36843, 195694, 194644, 187310, 39614]
mscore = [];
mtext = [];

print('Start ....');
for m in range(len(movie_code)):
    print('Movie Code:',movie_code[m]);
    for p in range(1,pages):
        print('start '+str(p)+' Page ...');
        html = requests.get('https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code='+str(movie_code[m])+
                            '&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false&page='+str(p))
        html_source = BeautifulSoup(html.content,'html.parser');
        i = 0;
        for li in html_source.find('div', {'class': 'score_result'}).find_all(('li')):
            reviews = html_source.find( 'span', {'id':'_filtered_ment_' + str(i)})
            if int(li.em.text) > 9:
                mscore.append(2); # 긍긍정
                mtext.append(reviews.text.strip());
            elif int(li.em.text) > 3:
                mscore.append(1);  # 긍정
                mtext.append(reviews.text.strip());
            elif int(li.em.text) <= 3:
                mscore.append(0);  # 부정
                mtext.append(reviews.text.strip());
            i += 1;

print('End ....');

# 네이버 영화에서 몇개의 영화를 선택 하여 한줄평 text 수집 한다.
# 수집된 TEXT를 파일로 저장 한다.

df = pd.DataFrame([mscore,mtext]).T;
df.columns = ['score','text'];
df['score'].value_counts();
df.to_excel('movies.xlsx',encoding='utf-8-sig');