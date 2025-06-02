import pandas as pd
from konlpy.tag import Mecab
from collections import Counter
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import os

# ===============================
# 설정
# ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")
model = BertForSequenceClassification.from_pretrained("beomi/kcbert-base")
model.to(device)
mecab = Mecab('C:/mecab/share/mecab-ko-dic')

# 불용어 (더 포괄적으로 확장)
stop_words = ['피부', '제품', '사용', '선물', '남자', '크림', '저녁', '아빠', '구매', '정도', '아침',
              '남성', '느낌', '효과', '토너', '추천', '이거', '가격', '마무리', '후기', '타입',
              '세일', '생각', '기획', '고민', '여름', '화장', '겨울', '남편', '만족', '올인원', '용량', '하나',
              '스킨', '마음', '처음', '리뷰', '로션', '입니다', '자극', '보습', '수분', '제형', '친구', '구입',
              '그냥', '완전', '진짜', '정말', '엄청', '너무', '좀', '조금', '약간', '살짝', '진짜로',
              '어요', '아요', '는데', '으로', '습니다', '해서', '라서', '에서', '까지', '부터', '으니까',
              '네요', '이에요', '예요', '이라', '라고', '다고', '해도', '써도', '했어요', '겠어요']

# 의미 있는 품사만 추출하기 위한 태그
meaningful_pos = ['NNG', 'NNP', 'NNB', 'VV', 'VA', 'MAG', 'MAJ']


# ===============================
# 함수 정의
# ===============================

def preprocess_text(text):
    """텍스트 전처리 - 문장 단위로 분리"""
    if not isinstance(text, str):
        return []

    # 특수문자 정리 (한글, 영문, 숫자, 공백, 기본 구두점만 남김)
    text = re.sub(r'[^\w\s,.!?ㄱ-ㅎㅏ-ㅣ가-힣]', ' ', text)

    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)

    # 문장 분리 (더 정확한 패턴)
    sentences = re.split(r'[.!?]+\s*', text)

    # 빈 문장이나 너무 짧은 문장 제거
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]

    return sentences


def extract_meaningful_tokens(text):
    """의미 있는 토큰만 추출 (품사 태깅 활용)"""
    if not isinstance(text, str) or len(text.strip()) < 2:
        return []

    try:
        # 품사 태깅
        pos_tags = mecab.pos(text)

        # 의미 있는 품사만 필터링
        meaningful_tokens = []
        for word, pos in pos_tags:
            # 길이 조건 + 품사 조건 + 불용어 제외
            if (len(word) >= 2 and
                    pos in meaningful_pos and
                    word not in stop_words and
                    not word.isdigit() and
                    not re.match(r'^[ㄱ-ㅎㅏ-ㅣ]+$', word)):  # 자음/모음만 있는 것 제외
                meaningful_tokens.append(word)

        return meaningful_tokens
    except:
        return []


def predict_sentiment(text):
    """감정 분석"""
    if not isinstance(text, str) or len(text.strip()) < 2:
        return -1

    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).cpu().numpy()[0]

        return pred
    except:
        return -1


def generate_meaningful_ngrams(tokens, n, min_count=2):
    """의미 있는 n-gram 생성"""
    if len(tokens) < n:
        return []

    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i:i + n])
        # 너무 짧거나 의미 없는 조합 제외
        if len(ngram) >= 4:  # 최소 길이 조건
            ngrams.append(ngram)

    return ngrams


def get_keyword_context(df, keyword, sentiment_filter=None):
    """키워드가 포함된 문장들의 컨텍스트 분석"""
    keyword_sentences = df[df['sentences'].str.contains(keyword, na=False, regex=False)]

    if sentiment_filter is not None:
        keyword_sentences = keyword_sentences[keyword_sentences['sentiment'] == sentiment_filter]

    return keyword_sentences


# ===============================
# 데이터 처리
# ===============================

print("데이터 로딩 중...")
df = pd.read_csv('./data/all_reviews.csv')

print("텍스트 전처리 중...")
df['sentences'] = df['review'].apply(preprocess_text)
df = df.explode('sentences').reset_index(drop=True)

# 빈 문장 제거
df = df[df['sentences'].notna() & (df['sentences'] != '')].reset_index(drop=True)

print("토큰 추출 중...")
df['tokens'] = df['sentences'].apply(extract_meaningful_tokens)

print("감정 분석 중...")
df['sentiment'] = df['sentences'].apply(predict_sentiment)

# 감정 분석 실패한 것들 제거
df = df[df['sentiment'] != -1].reset_index(drop=True)

print("N-gram 생성 중...")
df['bigrams'] = df['tokens'].apply(lambda x: generate_meaningful_ngrams(x, 2))
df['trigrams'] = df['tokens'].apply(lambda x: generate_meaningful_ngrams(x, 3))

# ===============================
# 출력 폴더 생성
# ===============================

os.makedirs("outputs", exist_ok=True)
markdown_lines = []
csv_rows = []

# ===============================
# 제품별 분석 (카테고리-제품 기준)
# ===============================

print("제품별 분석 시작...")
products = df[['category', 'product']].drop_duplicates()

for idx, row in products.iterrows():
    category = row['category']
    product = row['product']

    print(f"분석 중: {category} - {product}")

    product_df = df[(df['category'] == category) & (df['product'] == product)].copy()

    if len(product_df) < 5:  # 너무 적은 리뷰는 건너뛰기
        continue

    # 토큰 집계
    all_tokens = []
    for tokens in product_df['tokens']:
        all_tokens.extend(tokens)

    all_bigrams = []
    for bigrams in product_df['bigrams']:
        all_bigrams.extend(bigrams)

    all_trigrams = []
    for trigrams in product_df['trigrams']:
        all_trigrams.extend(trigrams)

    # 빈도 계산
    token_counts = Counter(all_tokens)
    bigram_counts = Counter(all_bigrams)
    trigram_counts = Counter(all_trigrams)

    # 상위 키워드 선정 (빈도가 높고 의미있는 것들)
    top_keywords = []
    for word, count in token_counts.most_common(50):
        if count >= 3 and len(word) >= 2:  # 최소 3번 이상 언급되고 2글자 이상
            top_keywords.append((word, count))
        if len(top_keywords) >= 10:  # 상위 10개까지
            break

    # 키워드별 감정 분석
    keyword_sentiment = {}
    for keyword, freq in top_keywords:
        keyword_sentences = get_keyword_context(product_df, keyword)

        if len(keyword_sentences) > 0:
            total = len(keyword_sentences)
            positive = len(keyword_sentences[keyword_sentences['sentiment'] == 1])
            negative = len(keyword_sentences[keyword_sentences['sentiment'] == 0])

            pos_rate = positive / total * 100
            keyword_sentiment[keyword] = {
                'positive_rate': pos_rate,
                'total_mentions': total,
                'positive_count': positive,
                'negative_count': negative
            }

    # 상위 구절 (빈도 2 이상)
    top_bigrams = [(phrase, count) for phrase, count in bigram_counts.most_common(10) if count >= 2]
    top_trigrams = [(phrase, count) for phrase, count in trigram_counts.most_common(10) if count >= 2]

    # 키워드별 대표 리뷰 추출
    keyword_examples = {}
    for keyword in keyword_sentiment.keys():
        pos_sentences = get_keyword_context(product_df, keyword, sentiment_filter=1)
        neg_sentences = get_keyword_context(product_df, keyword, sentiment_filter=0)

        pos_example = pos_sentences['sentences'].iloc[0] if len(pos_sentences) > 0 else "없음"
        neg_example = neg_sentences['sentences'].iloc[0] if len(neg_sentences) > 0 else "없음"

        keyword_examples[keyword] = (pos_example, neg_example)

    # 마크다운 출력 구성
    markdown_lines.append(f"\n\n## 📦 {category} | {product}")
    markdown_lines.append(f"*