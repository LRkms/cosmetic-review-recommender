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
model.eval()  # 평가 모드로 설정

# Mecab 초기화 개선 (Windows/Linux 호환)
try:
    mecab = Mecab('C:/mecab/share/mecab-ko-dic')
except:
    try:
        mecab = Mecab()
    except:
        print("Mecab 초기화 실패. 시스템에 Mecab이 설치되어 있는지 확인하세요.")
        exit(1)

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
    if not isinstance(text, str) or pd.isna(text):
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
    except Exception as e:
        print(f"토큰 추출 오류: {e}")
        return []


def predict_sentiment(text):
    """감정 분석 (배치 처리 최적화)"""
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

        return int(pred)  # numpy int를 Python int로 변환
    except Exception as e:
        print(f"감정 분석 오류: {e}")
        return -1


def predict_sentiment_batch(texts, batch_size=32):
    """배치 단위 감정 분석 (성능 개선)"""
    results = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        try:
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                results.extend([int(pred) for pred in preds])

        except Exception as e:
            print(f"배치 감정 분석 오류: {e}")
            results.extend([-1] * len(batch_texts))

    return results


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
    try:
        keyword_sentences = df[df['sentences'].str.contains(keyword, na=False, regex=False)]

        if sentiment_filter is not None:
            keyword_sentences = keyword_sentences[keyword_sentences['sentiment'] == sentiment_filter]

        return keyword_sentences
    except Exception as e:
        print(f"키워드 컨텍스트 분석 오류: {e}")
        return pd.DataFrame()


# ===============================
# 데이터 처리
# ===============================

print("데이터 로딩 중...")
try:
    df = pd.read_csv('./data/all_reviews.csv')
    print(f"총 {len(df)}개의 리뷰 로드 완료")
except FileNotFoundError:
    print("데이터 파일을 찾을 수 없습니다. './data/all_reviews.csv' 경로를 확인하세요.")
    exit(1)

print("텍스트 전처리 중...")
df['sentences'] = df['review'].apply(preprocess_text)
df = df.explode('sentences').reset_index(drop=True)

# 빈 문장 제거
df = df[df['sentences'].notna() & (df['sentences'] != '')].reset_index(drop=True)
print(f"전처리 후 {len(df)}개의 문장")

print("토큰 추출 중...")
df['tokens'] = df['sentences'].apply(extract_meaningful_tokens)

print("감정 분석 중... (배치 처리)")
# 배치 처리로 성능 개선
sentences_list = df['sentences'].tolist()
sentiments = predict_sentiment_batch(sentences_list)
df['sentiment'] = sentiments

# 감정 분석 실패한 것들 제거
df = df[df['sentiment'] != -1].reset_index(drop=True)
print(f"감정 분석 후 {len(df)}개의 문장 남음")

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
print(f"총 {len(products)}개 제품 분석 예정")

for idx, row in products.iterrows():
    category = row['category']
    product = row['product']

    print(f"분석 중 ({idx + 1}/{len(products)}): {category} - {product}")

    product_df = df[(df['category'] == category) & (df['product'] == product)].copy()

    if len(product_df) < 5:  # 너무 적은 리뷰는 건너뛰기
        print(f"  -> 리뷰가 {len(product_df)}개로 너무 적어 건너뛰기")
        continue

    # 토큰 집계
    all_tokens = []
    for tokens in product_df['tokens']:
        if isinstance(tokens, list):
            all_tokens.extend(tokens)

    all_bigrams = []
    for bigrams in product_df['bigrams']:
        if isinstance(bigrams, list):
            all_bigrams.extend(bigrams)

    all_trigrams = []
    for trigrams in product_df['trigrams']:
        if isinstance(trigrams, list):
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

            pos_rate = positive / total * 100 if total > 0 else 0
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
    markdown_lines.append(f"*총 리뷰 수: {len(product_df)}개*\n")

    # 전체 감정 분포
    total_reviews = len(product_df)
    positive_reviews = len(product_df[product_df['sentiment'] == 1])
    negative_reviews = len(product_df[product_df['sentiment'] == 0])
    pos_percentage = (positive_reviews / total_reviews * 100) if total_reviews > 0 else 0

    markdown_lines.append(f"### 📊 전체 감정 분석")
    markdown_lines.append(f"- 긍정: {positive_reviews}개 ({pos_percentage:.1f}%)")
    markdown_lines.append(f"- 부정: {negative_reviews}개 ({100 - pos_percentage:.1f}%)")

    # 키워드별 감정 분석 결과
    if keyword_sentiment:
        markdown_lines.append(f"\n### 🔍 주요 키워드 감정 분석")
        for keyword, stats in sorted(keyword_sentiment.items(), key=lambda x: x[1]['total_mentions'], reverse=True):
            pos_rate = stats['positive_rate']
            total = stats['total_mentions']
            pos_count = stats['positive_count']
            neg_count = stats['negative_count']

            sentiment_icon = "😊" if pos_rate >= 70 else "😐" if pos_rate >= 30 else "😞"

            markdown_lines.append(f"- **{keyword}** {sentiment_icon} (언급 {total}회)")
            markdown_lines.append(f"  - 긍정: {pos_count}회 ({pos_rate:.1f}%)")
            markdown_lines.append(f"  - 부정: {neg_count}회 ({100 - pos_rate:.1f}%)")

            # 대표 예시
            if keyword in keyword_examples:
                pos_ex, neg_ex = keyword_examples[keyword]
                if pos_ex != "없음":
                    markdown_lines.append(f"  - 긍정 예시: \"{pos_ex[:50]}...\"")
                if neg_ex != "없음":
                    markdown_lines.append(f"  - 부정 예시: \"{neg_ex[:50]}...\"")

    # 주요 구절
    if top_bigrams:
        markdown_lines.append(f"\n### 💬 자주 언급되는 구절")
        for phrase, count in top_bigrams[:5]:
            markdown_lines.append(f"- \"{phrase}\" ({count}회)")

    # CSV 데이터 추가
    for keyword, stats in keyword_sentiment.items():
        csv_rows.append({
            'category': category,
            'product': product,
            'keyword': keyword,
            'total_mentions': stats['total_mentions'],
            'positive_count': stats['positive_count'],
            'negative_count': stats['negative_count'],
            'positive_rate': stats['positive_rate']
        })

# ===============================
# 결과 저장
# ===============================

print("결과 저장 중...")

# 마크다운 파일 저장
markdown_content = "\n".join(markdown_lines)
with open("outputs/product_analysis.md", "w", encoding="utf-8") as f:
    f.write("# 🛍️ 제품별 리뷰 감정 분석 결과\n")
    f.write(markdown_content)

# CSV 파일 저장
if csv_rows:
    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv("outputs/keyword_sentiment_analysis.csv", index=False, encoding="utf-8-sig")

print("✅ 분석 완료!")
print(f"📁 결과 파일:")
print(f"  - outputs/product_analysis.md")
print(f"  - outputs/keyword_sentiment_analysis.csv")