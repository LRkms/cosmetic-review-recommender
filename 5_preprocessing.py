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

# 불용어
stop_words = ['피부', '제품', '사용', '선물', '남자', '크림', '저녁', '아빠', '구매', '정도', '아침',
              '남성', '느낌', '효과', '토너', '추천', '이거', '가격', '마무리', '후기', '타입',
              '세일', '생각', '기획', '고민', '여름', '화장', '겨울', '남편', '만족', '올인원', '용량', '하나',
              '스킨', '마음', '처음', '리뷰', '로션', '입니다', '자극', '보습', '수분', '제형', '친구', '구입']

# ===============================
# 함수 정의
# ===============================

def preprocess_text(text):
    if not isinstance(text, str):
        return []
    text = re.sub(r'[^\w\s,.!?]', '', text)
    for stop_word in stop_words:
        text = text.replace(stop_word, '')
    sentences = re.split('[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def extract_tokens(text):
    if not isinstance(text, str):
        return []
    return mecab.morphs(text)

def predict_sentiment(text):
    if not isinstance(text, str):
        return -1
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).cpu().numpy()[0]
    return pred

def generate_ngrams(tokens, n):
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# ===============================
# 데이터 처리
# ===============================

df = pd.read_csv('./data/all_reviews.csv')
df['sentences'] = df['review'].apply(preprocess_text)
df = df.explode('sentences')
df['tokens'] = df['sentences'].apply(extract_tokens)
df['sentiment'] = df['sentences'].apply(predict_sentiment)
df['bigrams'] = df['tokens'].apply(lambda x: generate_ngrams(x, 2))
df['trigrams'] = df['tokens'].apply(lambda x: generate_ngrams(x, 3))

# ===============================
# 출력 폴더 생성
# ===============================

os.makedirs("outputs", exist_ok=True)
markdown_lines = []
csv_rows = []

# ===============================
# 제품별 분석 (카테고리-제품 기준)
# ===============================

products = df[['category', 'product']].drop_duplicates()

for _, row in products.iterrows():
    category = row['category']
    product = row['product']
    product_df = df[(df['category'] == category) & (df['product'] == product)]

    all_tokens = [token for tokens in product_df['tokens'] for token in tokens if len(token) > 1]
    all_bigrams = [bigram for bigrams in product_df['bigrams'] for bigram in bigrams]
    all_trigrams = [trigram for trigrams in product_df['trigrams'] for trigram in trigrams]

    token_counts = Counter(all_tokens)
    bigram_counts = Counter(all_bigrams)
    trigram_counts = Counter(all_trigrams)

    top_keywords = [kw for kw in token_counts.most_common(20) if kw[0] not in ['어요', '아요', '는데', '으로', '습니다']][:5]
    top_bigrams = bigram_counts.most_common(5)
    top_trigrams = trigram_counts.most_common(5)

    keyword_sentiment = {}
    for keyword, _ in top_keywords:
        keyword_sentences = product_df[product_df['sentences'].str.contains(keyword, na=False)]
        if not keyword_sentences.empty:
            total = len(keyword_sentences)
            positive = len(keyword_sentences[keyword_sentences['sentiment'] == 1])
            keyword_sentiment[keyword] = f"{positive / total * 100:.0f}% 긍정"

    frequent_phrases = top_bigrams + top_trigrams

    summary = {}
    for keyword in keyword_sentiment.keys():
        keyword_sentences = product_df[product_df['sentences'].str.contains(keyword, na=False)]
        pos_reviews = keyword_sentences[keyword_sentences['sentiment'] == 1]['sentences'].head(1).iloc[0] if len(keyword_sentences[keyword_sentences['sentiment'] == 1]) > 0 else "없음"
        neg_reviews = keyword_sentences[keyword_sentences['sentiment'] == 0]['sentences'].head(1).iloc[0] if len(keyword_sentences[keyword_sentences['sentiment'] == 0]) > 0 else "없음"
        summary[keyword] = (pos_reviews, neg_reviews)

    # 출력 문자열 구성
    markdown_lines.append(f"\n\n## 📦 {category} | {product}")
    markdown_lines.append(f"\n### 🔑 주요 키워드 (긍정 비율):")
    for keyword, sentiment in keyword_sentiment.items():
        markdown_lines.append(f"- {keyword}: {sentiment}")

    markdown_lines.append(f"\n### 💬 자주 언급된 구절 (bi-gram/tri-gram):")
    for phrase, count in frequent_phrases:
        markdown_lines.append(f"- {phrase}: {count}회")

    markdown_lines.append(f"\n### 📝 리뷰 요약:")
    for keyword, (pos, neg) in summary.items():
        markdown_lines.append(f"- {keyword}: {pos} / {neg}")

    markdown_lines.append(f"\n### 🗣️ 상위 리뷰 문장 (빈도 기준):")
    top_sentences = product_df.groupby('sentences').size().sort_values(ascending=False).head(5).index
    for sentence in top_sentences:
        markdown_lines.append(f"- {sentence}")

    # CSV 저장용 row 추가
    for keyword, sentiment in keyword_sentiment.items():
        pos, neg = summary[keyword]
        csv_rows.append({
            "category": category,
            "product": product,
            "keyword": keyword,
            "positive_rate": sentiment,
            "positive_example": pos,
            "negative_example": neg
        })

# ===============================
# 결과 저장
# ===============================

with open("outputs/summary.md", "w", encoding="utf-8") as f:
    f.write("\n".join(markdown_lines))

pd.DataFrame(csv_rows).to_csv("outputs/summary.csv", index=False, encoding="utf-8-sig")
