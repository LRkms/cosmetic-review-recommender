import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.io import mmwrite, mmread
import pickle

# TF = 특정 문장안에 출현하는 단어 빈도수가 많은 거를 찾기
# DF =

df_reviews = pd.read_csv('./cleaned_data/cleaned_reviews.csv')
df_reviews.dropna(inplace=True)
df_reviews.info()

tfidf = TfidfVectorizer(sublinear_tf=True)
tfidf_matrix = tfidf.fit_transform(df_reviews['reviews'])
print(tfidf_matrix.shape)
print(tfidf_matrix[0])

with open('./models/tfidf.pickle', 'wb') as f:
    pickle.dump(tfidf, f)

mmwrite('./models/tfidf_cosmetic_review.mtx', tfidf_matrix)