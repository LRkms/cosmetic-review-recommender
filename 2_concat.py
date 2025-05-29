import pandas as pd
import glob
import os

# CSV 파일들이 있는 경로
data_paths = glob.glob('./data/*.csv')

# 최종 결과를 담을 DataFrame
df_all = pd.DataFrame(columns=['products', 'tags', 'reviews'])

for path in data_paths:
    df_temp = pd.read_csv(path)
    df_temp.columns = ['product', 'tag', 'review']  # 혹시 파일마다 순서 다를 때 대비

    # 중복된 제품명을 기준으로 리뷰/태그를 병합
    grouped = df_temp.groupby('product').agg({
        'tag': lambda x: ', '.join(x.dropna().astype(str)),      # 태그 병합
        'review': lambda x: ' '.join(x.dropna().astype(str))     # 리뷰 병합
    }).reset_index()

    # 열 이름 변경
    grouped.columns = ['products', 'tags', 'reviews']

    # 하나의 DataFrame에 누적
    df_all = pd.concat([df_all, grouped], ignore_index=True)

# 중복 제거 (같은 제품명에 대해 여러 파일에서 들어온 경우)
df_all.drop_duplicates(subset='products', inplace=True)

# 결과 저장
os.makedirs('./cleaned_data', exist_ok=True)
df_all.to_csv('./cleaned_data/cosmetic_reviews.csv', index=False, encoding='utf-8-sig')

# 확인 출력
print("✅ 통합 완료:", df_all.shape)
print(df_all.head(10))
