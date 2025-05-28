from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc
from datetime import datetime
import pickle
import time
import pandas as pd
import os

# ========================================
# 🎛️ 크롤링 설정
# ========================================
MAX_PAGES = 35
MAX_REVIEWS_PER_PRODUCT = 100
MAX_TAGS_PER_REVIEW = 5

PAGE_LOAD_WAIT = 3
PRODUCT_CLICK_WAIT = 2
REVIEW_TAB_WAIT = 2
BACK_WAIT = 2

UL_RANGE_START = 2
UL_RANGE_END = 8
LI_RANGE_START = 1
LI_RANGE_END = 5

HEADLESS_MODE = True
WINDOW_SIZE = "1920,1080"

# ========================================
# 🍀 Chrome Driver 설정 (undetected_chromedriver 사용)
# ========================================
options = uc.ChromeOptions()
options.add_argument("--lang=ko-KR")
options.add_argument("--no-sandbox")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument('--disable-gpu')
options.add_argument(f'--window-size={WINDOW_SIZE}')
user_agent = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
)
options.add_argument(f'--user-agent={user_agent}')

if HEADLESS_MODE:
    options.add_argument('--headless=new')  # Chrome 109 이상에서 권장

driver = uc.Chrome(options=options)
print("✅ 크롬 드라이버(undetected) 설정 완료")

# ========================================
# 🍪 쿠키 관련 함수
# ========================================
def load_cookies():
    if os.path.exists("cookies.pkl"):
        driver.get("https://kr.iherb.com")
        with open("cookies.pkl", "rb") as f:
            cookies = pickle.load(f)
            for cookie in cookies:
                driver.add_cookie(cookie)
        driver.refresh()
        time.sleep(3)

def save_cookies():
    with open("cookies.pkl", "wb") as f:
        pickle.dump(driver.get_cookies(), f)

if not os.path.exists("cookies.pkl"):
    print("❗ CAPTCHA 페이지가 보이면 수동으로 풀고 Enter를 누르세요...")
    driver.get("https://kr.iherb.com")
    input("👉 캡차를 통과했으면 Enter 키를 누르세요...")
    save_cookies()
else:
    load_cookies()

# ========================================
# 🗂️ 카테고리 설정
# ========================================
category_names = ['menscare']
prefixes = ['1000001000700']
subcategory_map = [[(7, 'toner')]]

os.makedirs('./data', exist_ok=True)

# ========================================
# 🔄 크롤링 루프 시작
# ========================================
total_start_time = time.time()
start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"🚀 크롤링 시작: {start_datetime}")
print(f"🎛️ 설정: 최대 {MAX_PAGES}페이지, 제품당 {MAX_REVIEWS_PER_PRODUCT}개 리뷰")

for idx in range(min(len(category_names), len(prefixes), len(subcategory_map))):
    category = category_names[idx]
    prefix = prefixes[idx]
    sub_list = subcategory_map[idx]

    category_start_time = time.time()

    for code, sub in sub_list:
        key = f"{category}_{sub}"
        category_data = []

        print(f"\n📁 [{category} → {sub}] 크롤링 시작")
        current_page = 1

        while current_page <= MAX_PAGES:
            page_url = (
                f'https://www.oliveyoung.co.kr/store/display/getMCategoryList.do?'
                f'dispCatNo={prefix}{code:02d}&fltDispCatNo=&prdSort=01&pageIdx={current_page}'
            )
            print(f"\n🌐 페이지 {current_page}/{MAX_PAGES} 접속: {page_url}")
            driver.get(page_url)
            time.sleep(PAGE_LOAD_WAIT)

            for ul_idx in range(UL_RANGE_START, UL_RANGE_END):
                for li_idx in range(LI_RANGE_START, LI_RANGE_END):
                    try:
                        xpath = (
                            f'//*[@id="Contents"]/ul[{ul_idx}]/li[{li_idx}]/div/div/a/p'
                        )
                        product_element = driver.find_element(By.XPATH, xpath)
                        name = product_element.text.strip()
                        print(f"    🔍 제품 발견: {name}")

                        driver.execute_script("arguments[0].click();", product_element)
                        time.sleep(PRODUCT_CLICK_WAIT)

                        try:
                            review_tab = WebDriverWait(driver, 5).until(
                                EC.element_to_be_clickable((By.CSS_SELECTOR, '#reviewInfo > a'))
                            )
                            review_tab.click()
                            time.sleep(REVIEW_TAB_WAIT)

                            try:
                                experience_checkbox = WebDriverWait(driver, 3).until(
                                    EC.element_to_be_clickable((By.CSS_SELECTOR, '#searchType div:nth-child(4) input'))
                                )
                                if experience_checkbox.is_selected():
                                    experience_checkbox.click()
                                    time.sleep(1)
                            except Exception:
                                pass

                            page_num = 1
                            reviews_collected = 0

                            while reviews_collected < MAX_REVIEWS_PER_PRODUCT:
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
                                                    f'//*[@id="gdasList"]/li[{r_idx}]/div[2]/div[2]/dl[{tag_idx}]/dd/span'
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
                                        print(f"        ✅ 리뷰 [{reviews_collected}/{MAX_REVIEWS_PER_PRODUCT}]: {review[:30]}...")

                                    except NoSuchElementException:
                                        continue

                                page_num += 1
                                try:
                                    btn_css = f'#gdasContentsArea > div > div.pageing > a:nth-child({page_num})'
                                    page_btn = driver.find_element(By.CSS_SELECTOR, btn_css)
                                    page_btn.click()
                                    time.sleep(REVIEW_TAB_WAIT)
                                    print(f"        ▶️ 리뷰 페이지 {page_num}로 이동")
                                except NoSuchElementException:
                                    break

                        except Exception as e:
                            print(f"      ❌ 리뷰 탭 클릭 또는 수집 오류: {e}")

                        driver.back()
                        time.sleep(BACK_WAIT)

                    except NoSuchElementException:
                        continue

            current_page += 1

        df = pd.DataFrame(category_data, columns=['product', 'tag', 'review'])
        df.to_csv(f'./data/{key}.csv', index=False, encoding='utf-8-sig')

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

print("\n🛑 브라우저 종료 중...")
driver.quit()