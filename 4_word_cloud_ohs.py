import pandas as pd
from wordcloud import WordCloud
import collections
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = './malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rc('font', family='NanumBarunGothic')

stop_words = ['리뷰','도움이','돼요','이에요','좋아요','순해요','만족해요','보통',
                  '제품','사용','합니다','있다','없다']

df = pd.read_csv('./cleaned_data/cleaned_reviews.csv')
print(len(df))
print(df.head())

sentence = df.iloc[-1]['reviews']
for stop_word in stop_words:
    sentence = sentence.replace(stop_word, '')
print(sentence)
words = sentence.split()


# 각 형태소의 출현 쵯수를 딕셔너리로 묶어서
worddict = collections.Counter(words)
worddict = dict(worddict)
print(worddict)

# 각 단어의 출현 횟수를 이미지로 표현
wordcloud_img = WordCloud(
    background_color='white', max_words=2000, font_path=font_path
    ).generate_from_frequencies(worddict)
plt.figure(figsize=(12, 12))
plt.imshow(wordcloud_img, interpolation='bilinear')
plt.axis("off")
plt.show()