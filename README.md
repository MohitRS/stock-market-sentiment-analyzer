# 📈 Stock Market Sentiment-Based Prediction

## 🚀 Project Overview
This project provides an interactive **Stock Market Prediction App** that forecasts ETF closing prices based on:
- 📊 **Market sentiment analysis** (from financial news)
- 📈 **Previous day's closing price**
- 🔄 **Daily return percentage**

The app uses **machine learning models** trained on historical market data and sentiment scores to provide accurate predictions.

---

## 📂 Folder Structure
```
stock-market-sentiment-analyzer/
│-- data/                  # Contains historical financial & sentiment data
│-- models/                # Stores trained models and scalers
│-- src/                   # Source code for data processing, model training, and app
│   │-- fetch_yfinance_data.py      # Fetches financial data from Yahoo Finance
│   │-- fetch_news_sentiment.py     # Pulls sentiment data from news sources
│   │-- predictive_model.py         # Trains multiple ML models & selects the best
│   │-- app.py                      # Streamlit web app
│-- README.md               # Project documentation
│-- requirements.txt        # Dependencies
```

---

## ⚙️ Installation Guide
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/stock-market-sentiment-analyzer.git
cd stock-market-sentiment-analyzer
```

### 2️⃣ Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate    # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Use
### 1️⃣ Train the Model
Run the following to download financial and sentiment data, then train the model:
```bash
python src/fetch_yfinance_data.py   # Fetches stock price data
python src/fetch_news_sentiment.py  # Fetches sentiment data
python src/predictive_model.py      # Trains and saves the best model
```

### 2️⃣ Run the Streamlit Web App
```bash
streamlit run src/app.py
```
Then open **`http://localhost:8501`** in your browser.

---

## 🏗 Machine Learning Models Used
The project tests multiple regression models and selects the best one based on:
✅ **Mean Absolute Error (MAE)**  
✅ **R² Score**  
✅ **Cross-Validation Score (CV MAE)**  

### 🔥 **Final Model Selected**: `HuberRegressor`
This model was chosen for its robustness against outliers and overfitting.

---

## 🔧 Future Enhancements
- ✅ **Deploy the model online (Streamlit Cloud / Hugging Face Spaces)**
- 📊 **Use real-time news & stock data for dynamic updates**
- 🏗 **Experiment with deep learning models (LSTMs, Transformers)**
- 📉 **Improve interpretability using SHAP analysis**

---

## 💡 Contributors & Credits
👤 **Mohit Rudraraju Suresh**  
📬 [LinkedIn](https://www.linkedin.com/in/mohitrs)  
📧 Email: mohitrs53@gmail.com  

**📌 Star the repo if you find it useful! ⭐**

