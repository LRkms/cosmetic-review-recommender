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


# -------------------------------
# 🍀 Chrome Driver 설정
# -------------------------------
options = ChromeOptions()
# 브라우저에서 자동화 탐지 방지
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
# 사용자 에이전트 및 창 크기 설정
user_agent = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
)
options.add_argument(f'--user-agent={user_agent}')
options.add_argument('--window-size=1920,1080')

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
# -------------------------------
# 📦 카테고리 설정
# -------------------------------
category_names = ['skincare', 'cleansing', 'suncare', 'menscare']
prefixes = [
    '1000001000100',  # 스킨케어
    # '1000001001000',  # 클렌징
    # '1000001001100',  # 선케어
    # '1000001000700',  # 맨즈케어
]
# 각 카테고리별 하위 카테고리 코드 및 키 이름
subcategory_map = [
    # [(13, 'toner'), (14, 'serum'), (15, 'cream'), (16, 'lotion'), (10, 'mist_oil')],
    # [(1, 'foam_gel'), (4, 'oil_balm'), (5, 'water_milk'), (7, 'peeling_scrub')],
    # [(6, 'suncream'), (3, 'sunstick'), (4, 'suncushion'), (5, 'sunspray_patch')],
    [(7, 'toner')],
]

# 데이터 저장 디렉토리 생성
os.makedirs('./data', exist_ok=True)

# -------------------------------
# 🔄 크롤링 루프
# -------------------------------
for idx in range(len(category_names)):
    category = category_names[idx]
    prefix = prefixes[idx]
    sub_list = subcategory_map[idx]

    for code, sub in sub_list:
        key = f"{category}_{sub}"
        category_data = []  # 해당 서브카테고리 리뷰 저장

        print(f"\n📁 [{category} → {sub}] 크롤링 시작")
        current_page = 1
        MAX_PAGE = 1  # 테스트용으로 3페이지만 크롤링

        # 페이지별 반복
        while current_page <= MAX_PAGE:
            page_url = (
                f'https://www.oliveyoung.co.kr/store/display/getMCategoryList.do?'
                f'dispCatNo={prefix}{code:02d}&fltDispCatNo=&prdSort=01&pageIdx={current_page}'
            )
            print(f"\n🌐 페이지 {current_page} 접속: {page_url}")
            driver.get(page_url)
            time.sleep(3)  # 페이지 로딩 대기

            # ul[2]~ul[7], li[1]~li[4] 내 제품 탐색
            for ul_idx in range(2, 8):
                for li_idx in range(1, 5):
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
                        time.sleep(2)

                        # -------------------------------
                        # 💬 리뷰 탭 클릭 (CSS 방식)
                        # -------------------------------
                        try:
                            review_tab = WebDriverWait(driver, 5).until(
                                EC.element_to_be_clickable((By.CSS_SELECTOR, '#reviewInfo > a'))
                            )
                            review_tab.click()
                            time.sleep(2)
                            try:
                                print("        🔄 체험단 필터 해제 중...")
                                # 방법 1: 체크박스 직접 클릭
                                experience_checkbox = WebDriverWait(driver, 3).until(
                                    EC.element_to_be_clickable((By.CSS_SELECTOR, '#searchType div:nth-child(4) input'))
                                )
                                if experience_checkbox.is_selected():  # 체크되어 있으면
                                    experience_checkbox.click()  # 클릭해서 해제
                                    time.sleep(1)
                                    print("        ✅ 체험단 필터 해제 완료")
                            except Exception as e:
                                print(f"        ⚠️ 체험단 필터 해제 실패 (계속 진행): {e}")
                            # 최대 5개 리뷰 수집
                            for r_idx in range(1, 6):
                                try:
                                    # 리뷰 텍스트 추출
                                    review_xpath = (
                                        f'//*[@id="gdasList"]/li[{r_idx}]/div[2]/div[3]'
                                    )
                                    review = driver.find_element(By.XPATH, review_xpath).text.strip()
                                    # 태그 추출
                                    tags = []
                                    for tag_idx in range(1, 6):
                                        try:
                                            tag_xpath = (
                                                f'//*[@id="gdasList"]/li[{r_idx}]/'
                                                f'div[2]/div[2]/dl[{tag_idx}]/dd/span'
                                            )
                                            tag = driver.find_element(By.XPATH, tag_xpath).text.strip()
                                            tags.append(tag)
                                        except NoSuchElementException:
                                            continue

                                    # 데이터 저장
                                    category_data.append({
                                        'product': name,
                                        'tag': ', '.join(tags),
                                        'review': review
                                    })
                                    # 🏷️ 태그 실시간 출력
                                    print(f"        🏷️ 태그: {tags}")
                                    print(f"        ✅ 리뷰 수집 완료 [li={r_idx}]: {review[:30]}...")
                                except NoSuchElementException:
                                    continue
                        except Exception as e:
                            print(f"      ❌ 리뷰 탭 클릭 실패: {e}")

                        # 상세 → 목록으로 돌아가기
                        driver.back()
                        time.sleep(2)

                    except NoSuchElementException:
                        continue

            current_page += 1

        # -------------------------------
        # 💾 결과 저장
        # -------------------------------
        df = pd.DataFrame(category_data, columns=['product', 'tag', 'review'])
        df.to_csv(f'./data/{key}.csv', index=False, encoding='utf-8-sig')
        print(f"✅ 저장 완료: {key}.csv (총 {len(category_data)}개 리뷰)")

total_end_time = time.time()
total_duration = total_end_time - total_start_time
end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"\n🏁 크롤링 완료: {end_datetime}")
print(f"⏱️ 총 소요시간: {total_duration:.1f}초 ({total_duration/60:.1f}분)")
if total_duration >= 3600:
    print(f"⏱️ 총 소요시간: {total_duration/3600:.1f}시간")

# 브라우저 종료
print("\n🛑 브라우저 종료 중...")
driver.quit()