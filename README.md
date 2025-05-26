# cosmetic-review-recommender
# 💄 Cosmetic Review-Based Recommendation System

A personalized cosmetic recommendation system based on user reviews, designed to suggest skincare products by **gender** and **skin type**.  
The application includes a **PyQt5-based GUI** where users can select their skin characteristics and receive product suggestions tailored to their needs.

---

## 🧠 Features

- ✅ Collects real-world cosmetic product reviews
- ✅ Analyzes text data using NLP (Word2Vec / TF-IDF)
- ✅ Recommends products based on:
  - Gender
  - Skin type (e.g. dry, oily, sensitive, combination)
  - User preference keywords (optional)
- ✅ Built-in PyQt5 GUI for user-friendly experience

---

## 🖥 Application Preview

> 

---

## 🧩 System Components

| Component | Description |
|----------|-------------|
| `scraper/` | Web crawler to collect product reviews from Olive Young |
| `preprocessing/` | Cleans and tokenizes review text (Korean NLP) |
| `models/` | Trained Word2Vec and TF-IDF models |
| `app/` | PyQt5-based GUI application |
| `data/` | Processed datasets (product info, labels) |

---

## 🛠 Technologies Used

- Python 3.x  
- PyQt5  
- Pandas, NumPy  
- Gensim (Word2Vec)  
- Scikit-learn (TF-IDF, cosine similarity)  
- BeautifulSoup / Selenium (for scraping)  
- Mecab-ko or Okt (Korean NLP)

---

## 🚀 How to Run

```bash
# 1. Clone this repository
git clone https://github.com/your-username/your-repo-name.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the PyQt5 app
python app/main.py
