from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime
import time
import pandas as pd
import os


# ========================================
# 🎛️ 크롤링 설정 (테스트/운영 쉽게 변경)
# ========================================
# 페이지 설정
MAX_PAGES = 1  # 크롤링할 최대 페이지 수 (테스트: 1, 운영: 10+)
MAX_REVIEWS_PER_PRODUCT = 100  # 제품당 수집할 최대 리뷰 수 (테스트: 5, 운영: 20+)
MAX_TAGS_PER_REVIEW = 5  # 리뷰당 수집할 최대 태그 수

# 대기 시간 설정 (초)
PAGE_LOAD_WAIT = 3  # 페이지 로딩 대기 시간
PRODUCT_CLICK_WAIT = 2  # 제품 클릭 후 대기 시간
REVIEW_TAB_WAIT = 2  # 리뷰 탭 클릭 후 대기 시간
BACK_WAIT = 2 # 뒤로가기 후 대기 시간

# 제품 탐색 범위 설정
UL_RANGE_START = 2  # ul 탐색 시작 인덱스
UL_RANGE_END = 8  # ul 탐색 종료 인덱스 (exclusive)
LI_RANGE_START = 1  # li 탐색 시작 인덱스
LI_RANGE_END = 5  # li 탐색 종료 인덱스 (exclusive)

# 브라우저 설정
HEADLESS_MODE = True  # True: 백그라운드 실행, False: 브라우저 화면 표시
WINDOW_SIZE = "1920,1080"  # 브라우저 창 크기

# ========================================
# 🍀 Chrome Driver 설정
# ========================================
options = ChromeOptions()
# 브라우저에서 자동화 탐지 방지
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)

# 헤드리스 모드 설정
if HEADLESS_MODE:
    options.add_argument('--headless')
    # 헤드리스 모드 최적화 옵션들
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-images')  # 이미지 로딩 안함
    options.add_argument('--disable-javascript')  # JS 일부 비활성화
    print("🚀 헤드리스 모드로 실행 - 리소스 최적화 적용")

# 사용자 에이전트 및 창 크기 설정
user_agent = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
)
options.add_argument(f'--user-agent={user_agent}')
options.add_argument(f'--window-size={WINDOW_SIZE}')

# webdriver-manager로 드라이버 자동 설치 및 실행
driver = webdriver.Chrome(
    service=ChromeService(ChromeDriverManager().install()),
    options=options
)
# navigator.webdriver 속성 변경으로 탐지 방지
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
print("✅ 크롬 드라이버 설정 완료")

total_start_time = time.time()
start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"🚀 크롤링 시작: {start_datetime}")
print(f"🎛️ 설정: 최대 {MAX_PAGES}페이지, 제품당 {MAX_REVIEWS_PER_PRODUCT}개 리뷰")

# ========================================
# 📦 카테고리 설정
# ========================================
category_names = ['skincare', 'cleansing', 'suncare', 'menscare']
prefixes = [
    '1000001000100',  # 스킨케어
    # '1000001001000',  # 클렌징
    # '1000001001100',  # 선케어
    # '1000001000700',  # 맨즈케어
]
# 각 카테고리별 하위 카테고리 코드 및 키 이름
subcategory_map = [
    [(13, 'toner')],  # (14, 'serum'), (15, 'cream'), (16, 'lotion'), (10, 'mist_oil')],
    # [(1, 'foam_gel'), (4, 'oil_balm'), (5, 'water_milk'), (7, 'peeling_scrub')],
    # [(6, 'suncream'), (3, 'sunstick'), (4, 'suncushion'), (5, 'sunspray_patch')],
    # [(7, 'toner')]
]

# 데이터 저장 디렉토리 생성
os.makedirs('./data', exist_ok=True)

# ========================================
# 🔄 크롤링 루프
# ========================================
for idx in range(min(len(category_names), len(prefixes), len(subcategory_map))):
    category = category_names[idx]
    prefix = prefixes[idx]
    sub_list = subcategory_map[idx]

    # 카테고리별 시작 시간 기록
    category_start_time = time.time()

    for code, sub in sub_list:
        key = f"{category}_{sub}"
        category_data = []  # 해당 서브카테고리 리뷰 저장

        print(f"\n📁 [{category} → {sub}] 크롤링 시작")
        current_page = 1

        # 페이지별 반복
        while current_page <= MAX_PAGES:
            page_url = (
                f'https://www.oliveyoung.co.kr/store/display/getMCategoryList.do?'
                f'dispCatNo={prefix}{code:02d}&fltDispCatNo=&prdSort=01&pageIdx={current_page}'
            )
            print(f"\n🌐 페이지 {current_page}/{MAX_PAGES} 접속: {page_url}")
            driver.get(page_url)
            time.sleep(PAGE_LOAD_WAIT)  # 페이지 로딩 대기

            # ul[2]~ul[7], li[1]~li[4] 내 제품 탐색
            for ul_idx in range(UL_RANGE_START, UL_RANGE_END):
                for li_idx in range(LI_RANGE_START, LI_RANGE_END):
                    try:
                        # 제품 요소 찾기 및 이름 추출
                        xpath = (
                            f'//*[@id="Contents"]/ul[{ul_idx}]/li[{li_idx}]/'
                            'div/div/a/p'
                        )
                        product_element = driver.find_element(By.XPATH, xpath)
                        name = product_element.text.strip()
                        print(f"    🔍 제품 발견: {name}")

                        # 제품 상세 페이지로 이동
                        driver.execute_script("arguments[0].click();", product_element)
                        time.sleep(PRODUCT_CLICK_WAIT)

                        # -------------------------------
                        # 💬 리뷰 탭 클릭 (CSS 방식)
                        # -------------------------------
                        try:
                            review_tab = WebDriverWait(driver, 5).until(
                                EC.element_to_be_clickable((By.CSS_SELECTOR, '#reviewInfo > a'))
                            )
                            review_tab.click()
                            time.sleep(REVIEW_TAB_WAIT)

                            # 체험단 필터 해제 (기존 코드)
                            try:
                                print("        🔄 체험단 필터 해제 중...")
                                experience_checkbox = WebDriverWait(driver, 3).until(
                                    EC.element_to_be_clickable((By.CSS_SELECTOR, '#searchType div:nth-child(4) input'))
                                )
                                if experience_checkbox.is_selected():
                                    experience_checkbox.click()
                                    time.sleep(1)
                                    print("        ✅ 체험단 필터 해제 완료")
                            except Exception as e:
                                print(f"        ⚠️ 체험단 필터 해제 실패 (계속 진행): {e}")

                            # ➡️ 리뷰 페이지별 수집 시작
                            page_num = 1
                            reviews_collected = 0
                            while reviews_collected < MAX_REVIEWS_PER_PRODUCT:
                                # 현재 페이지 리뷰 수집
                                for r_idx in range(1, MAX_REVIEWS_PER_PRODUCT + 1):
                                    if reviews_collected >= MAX_REVIEWS_PER_PRODUCT:
                                        break
                                    try:
                                        review_xpath = f'//*[@id="gdasList"]/li[{r_idx}]/div[2]/div[3]'
                                        review = driver.find_element(By.XPATH, review_xpath).text.strip()

                                        tags = []
                                        for tag_idx in range(1, MAX_TAGS_PER_REVIEW + 1):
                                            try:
                                                tag_xpath = (
                                                    f'//*[@id="gdasList"]/li[{r_idx}]/'
                                                    f'div[2]/div[2]/dl[{tag_idx}]/dd/span'
                                                )
                                                tag = driver.find_element(By.XPATH, tag_xpath).text.strip()
                                                tags.append(tag)
                                            except NoSuchElementException:
                                                continue

                                        category_data.append({
                                            'product': name,
                                            'tag': ', '.join(tags),
                                            'review': review
                                        })
                                        reviews_collected += 1
                                        print(f"        🏷️ 태그: {tags}")
                                        print(
                                            f"        ✅ 리뷰 [{reviews_collected}/{MAX_REVIEWS_PER_PRODUCT}]: {review[:30]}...")
                                    except NoSuchElementException:
                                        continue

                                # 다음 페이지 버튼 클릭
                                page_num += 1
                                try:
                                    btn_css = f'#gdasContentsArea > div > div.pageing > a:nth-child({page_num})'
                                    page_btn = driver.find_element(By.CSS_SELECTOR, btn_css)
                                    page_btn.click()
                                    time.sleep(REVIEW_TAB_WAIT)
                                    print(f"        ▶️ 리뷰 페이지 {page_num}로 이동")
                                except NoSuchElementException:
                                    print(f"        🚫 {page_num}번 페이지 버튼 못 찾았어, 리뷰 수집 종료")
                                    break

                        except Exception as e:
                            print(f"      ❌ 리뷰 탭 클릭 또는 수집 오류: {e}")

                        # 상세 → 목록으로 돌아가기
                        driver.back()
                        time.sleep(BACK_WAIT)

                    except NoSuchElementException:
                        continue

            current_page += 1

        # -------------------------------
        # 💾 결과 저장
        # -------------------------------
        df = pd.DataFrame(category_data, columns=['product', 'tag', 'review'])
        df.to_csv(f'./data/{key}.csv', index=False, encoding='utf-8-sig')

        # 카테고리별 소요 시간 계산
        category_end_time = time.time()
        category_duration = category_end_time - category_start_time

        print(f"✅ 저장 완료: {key}.csv (총 {len(category_data)}개 리뷰)")
        print(f"⏱️ {key} 소요시간: {category_duration:.1f}초 ({category_duration / 60:.1f}분)")

# ========================================
# 📊 최종 결과 출력
# ========================================
total_end_time = time.time()
total_duration = total_end_time - total_start_time
end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"\n🏁 크롤링 완료: {end_datetime}")
print(f"⏱️ 총 소요시간: {total_duration:.1f}초 ({total_duration / 60:.1f}분)")
if total_duration >= 3600:
    print(f"⏱️ 총 소요시간: {total_duration / 3600:.1f}시간")

# 브라우저 종료
print("\n🛑 브라우저 종료 중...")
driver.quit()