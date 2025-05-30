import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
import pickle
from konlpy.tag import Okt
from gensim.models import Word2Vec
import argparse

# --------------------
# 리소스 로딩
# --------------------
df_reviews = pd.read_csv('./cleaned_data/cleaned_reviews.csv')
tfidf_matrix = mmread('./models/tfidf_cosmetic_review.mtx').tocsr()
with open('./models/tfidf.pickle', 'rb') as f:
    tfidf = pickle.load(f)
embedding_model = Word2Vec.load('./models/word2vec_cosmetic_review.model')
okt = Okt()

# --------------------
# 추천 로직
# --------------------
def get_recommendations(cosine_sim, top_n=10):
    """
    코사인 유사도 행렬에서 상위 top_n개의 인덱스 기반 추천 목록을 반환합니다.
    """
    scores = list(enumerate(cosine_sim[-1]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[: top_n]
    indices = [idx for idx, _ in scores]
    rec_items = df_reviews.iloc[indices, 0]
    return rec_items.reset_index(drop=True)


def generate_keywords(raw_keyword):
    """
    입력 키워드를 형태소 분석 후, 워드투벡 모델 어휘에서 유효한 토큰을 기반으로
    유사 단어를 확장하여 반환합니다.
    """
    tokens = okt.morphs(raw_keyword)
    valid_tokens = [t for t in tokens if t in embedding_model.wv]

    all_similar = []
    for token in valid_tokens:
        try:
            sims = embedding_model.wv.most_similar(token, topn=5)
            all_similar.extend([w for w, _ in sims])
        except KeyError:
            continue
    unique_similar = list(dict.fromkeys(all_similar))

    for alt in valid_tokens + ['로션', 'AHC', '포맨', '남성']:
        if alt in embedding_model.wv:
            selected = alt
            break
    else:
        selected = None

    if selected:
        try:
            sims = embedding_model.wv.most_similar(selected, topn=10)
            words = [selected] + [w for w, _ in sims]
        except KeyError:
            words = valid_tokens
    else:
        words = valid_tokens

    return words


def recommend(keyword, top_n=10):
    """
    주어진 키워드에 대해 추천 제품 리스트를 반환합니다.
    """
    words = generate_keywords(keyword)
    if not words:
        return []

    sentence = []
    count = top_n
    for w in words:
        sentence += [w] * count
        count = max(1, count - 1)
    sentence_str = ' '.join(sentence)

    sentence_vec = tfidf.transform([sentence_str])
    cosine_sim = linear_kernel(sentence_vec, tfidf_matrix)

    return get_recommendations(cosine_sim, top_n=top_n)


def main():
    parser = argparse.ArgumentParser(
        description="Cosmetic Review Recommendation System"
    )
    parser.add_argument('keyword', nargs='?', help='추천에 사용할 키워드 문자열')
    parser.add_argument('--top_n', type=int, default=10, help='추천 제품 개수')
    args = parser.parse_args()

    if not args.keyword:
        args.keyword = input('추천에 사용할 키워드를 입력하세요: ').strip()
        if not args.keyword:
            print('키워드를 입력하지 않아 프로그램을 종료합니다.')
            return

    results = recommend(args.keyword, top_n=args.top_n)
    print("추천 제품들:")
    if not results.empty:
        for idx, item in enumerate(results, 1):
            print(f"{idx}. {item}")
    else:
        print('추천 결과가 없습니다.')

if __name__ == "__main__":
    main()


# import pandas as pd
# from sklearn.metrics.pairwise import linear_kernel
# from scipy.io import mmread
# import pickle
# from konlpy.tag import Okt
# import re
# import datetime
# from gensim.models import Word2Vec
#
#
# # 필요한 모델들과 데이터 로드
# def getRecommendations(cosine_sim):
#     simScore = list(enumerate(cosine_sim[-1]))
#     simScore = sorted(simScore, key=lambda x: x[1], reverse=True)
#     simScore = simScore[:11]
#     movie_idx = [i[0] for i in simScore]
#     rec_movie_list = df_reviews.iloc[movie_idx, 0]
#     return rec_movie_list[:11]
#
#
# df_reviews = pd.read_csv('./cleaned_data/cleaned_reviews.csv')
# df_reviews.info()
# tfidf_matrix = mmread('./models/tfidf_cosmetic_review.mtx').tocsr()
#
# with open('./models/tfidf.pickle', 'rb') as f:
#     tfidf = pickle.load(f)
#
# # Word2Vec 모델 로드 (이 부분이 빠져있었습니다!)
# embedding_model = Word2Vec.load('./models/word2vec_cosmetic_review.model')
#
# # 이제 토큰화 및 키워드 처리
# okt = Okt()
# keyword = '건성 남자 올인원'
#
# # 키워드를 토큰화
# tokens = okt.morphs(keyword)
# print("토큰화된 키워드:", tokens)
#
# # 모델에 존재하는 토큰만 필터링
# valid_tokens = []
# for token in tokens:
#     if token in embedding_model.wv:
#         valid_tokens.append(token)
#
# print("모델에 있는 토큰들:", valid_tokens)
#
# # 유효한 토큰들로 유사 단어 찾기
# all_similar_words = []
# for token in valid_tokens:
#     try:
#         sim_words = embedding_model.wv.most_similar(token, topn=5)
#         all_similar_words.extend([word for word, score in sim_words])
#     except KeyError:
#         continue
#
# # 중복 제거
# unique_words = list(set(all_similar_words))[:10]
#
# # 모델의 어휘에 있는 키워드들 확인
# vocab_sample = list(embedding_model.wv.key_to_index.keys())[:20]
# print("모델 어휘 예시:", vocab_sample)
#
# # 유사한 키워드 직접 선택
# alternative_keywords = ['로션', 'AHC', '포맨', '남성']
# selected_keyword = None
# for alt_keyword in alternative_keywords:
#     if alt_keyword in embedding_model.wv:
#         selected_keyword = alt_keyword
#         print(f"사용할 키워드: {selected_keyword}")
#         break
#
# # 최종적으로 사용할 단어들 결정
# words = []
# if selected_keyword:
#     try:
#         sim_word = embedding_model.wv.most_similar(selected_keyword, topn=10)
#         words = [selected_keyword]
#         for word, _ in sim_word:
#             words.append(word)
#     except KeyError as e:
#         print(f"키워드 '{selected_keyword}'가 모델 어휘에 없습니다.")
#         words = valid_tokens  # 유효한 토큰들만 사용
# else:
#     print("대체 키워드를 찾을 수 없습니다. 토큰화된 단어들을 사용합니다.")
#     words = valid_tokens
#
# # 문장 생성 및 추천
# if words:
#     sentence = []
#     count = 10
#     for word in words:
#         sentence = sentence + [word] * count
#         count = max(1, count - 1)  # count가 0이 되지 않도록
#
#     sentence = ' '.join(sentence)
#     print("생성된 문장:", sentence)
#
#     # TF-IDF 벡터화 및 추천
#     sentence_vec = tfidf.transform([sentence])
#     cosine_sim = linear_kernel(sentence_vec, tfidf_matrix)
#     recommendations = getRecommendations(cosine_sim)
#     recommendations = recommendations.reset_index(drop=True)
#     print("추천 제품들:")
#     print(recommendations[:10])
# else:
#     print("사용할 수 있는 키워드가 없습니다.")

