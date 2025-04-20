import pandas as pd
import numpy as np
import logging
import requests
from bs4 import BeautifulSoup
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AIPredictor:
    def __init__(self, alpha_vantage_key):
        self.alpha_vantage_key = alpha_vantage_key
        self.tokenizer = None
        self.model = None

    def initialize_finbert(self):
        """Initialize FinBERT for sentiment analysis"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            logging.info("FinBERT initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing FinBERT: {e}")
            self.tokenizer, self.model = None, None

    def fetch_technical_data(self, ticker):
        """Fetch technical data using Alpha Vantage"""
        try:
            ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
            ti = TechIndicators(key=self.alpha_vantage_key, output_format='pandas')
            data, _ = ts.get_daily(symbol=ticker, outputsize='compact')
            data = data.sort_index()
            sma200, _ = ti.get_sma(symbol=ticker, interval='daily', time_period=200)
            time.sleep(12)  # Respect Alpha Vantage rate limit (5 calls/min)
            return {
                'sma200': sma200['SMA'].iloc[-1] if not sma200.empty else data['4. close'].iloc[-1] * 0.97,
                'recent_low': data['3. low'].tail(30).min()  # 30-day low
            }
        except Exception as e:
            logging.error(f"Error fetching technical data for {ticker}: {e}")
            return {'sma200': 0, 'recent_low': 0}

    def fetch_sentiment(self, ticker):
        """Fetch financial news sentiment using FinBERT"""
        if not self.tokenizer or not self.model:
            self.initialize_finbert()
        try:
            # Mock news headlines (replace with real news API)
            headlines = {
                'NXPI': [
                    "NXP Semiconductors reports strong Q4 2024 earnings but issues soft Q1 2025 guidance due to automotive demand.",
                    "Semiconductor industry poised for rebound in 2025, boosting NXPI’s IoT and automotive segments.",
                    "NXPI faces challenges in industrial IoT amid global supply chain constraints."
                ],
                'HDFCBANK.NS': [
                    "HDFC Bank posts stable Q4 FY25 results with 2.2% YoY profit growth, maintaining strong asset quality.",
                    "Indian banking sector expected to benefit from RBI’s rate stabilization, favoring HDFC Bank.",
                    "HDFC Bank faces margin pressure due to rising deposit costs in FY25."
                ]
            }.get(ticker, ["Neutral news for stock analysis."])
            sentiments = []
            for headline in headlines:
                inputs = self.tokenizer(headline, return_tensors="pt", truncation=True, max_length=512)
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
                sentiment_score = probs[1] - probs[0]  # Positive - Negative
                sentiments.append(sentiment_score)
            avg_sentiment = np.mean(sentiments)
            sentiment_label = "Positive" if avg_sentiment > 0 else "Neutral" if avg_sentiment == 0 else "Negative"
            return sentiment_label, avg_sentiment
        except Exception as e:
            logging.error(f"Error fetching sentiment for {ticker}: {e}")
            return "Neutral", 0.0

    def fetch_analyst_target(self, ticker, current_price):
        """Fetch analyst price target (mocked, replace with scraping/API)"""
        try:
            # Mock targets based on previous data
            targets = {
                'NXPI': 263.52,  # Zacks, 27 analysts, April 2025
                'HDFCBANK.NS': 2304.00  # ~20% upside from ₹1920, analyst consensus
            }
            target_price = targets.get(ticker, current_price * 1.2)
            return target_price
        except Exception as e:
            logging.error(f"Error fetching analyst target for {ticker}: {e}")
            return current_price * 1.2

    def calculate_ai_predictions(self, row):
        """Calculate detailed AI-driven predictions"""
        ticker = row['Ticker']
        current_price = row['Current Price']
        volatility = row['Volatility']
        rsi = row['RSI']
        composite_score = row['Composite Score']
        de_ratio = row.get('D/E Ratio', 1.0)

        # Fetch technical data
        tech_data = self.fetch_technical_data(ticker)
        sma200 = tech_data['sma200']
        recent_low = tech_data['recent_low']

        # Fetch sentiment
        sentiment_label, avg_sentiment = self.fetch_sentiment(ticker)

        # Fetch analyst target
        target_price = self.fetch_analyst_target(ticker, current_price)

        # Support Zone: Based on 200-day SMA and recent low
        support_low = min(sma200, recent_low, current_price * 0.97)
        support_high = support_low + (5.0 if ticker == 'NXPI' else 50.0)  # $5 for NXPI, ₹50 for HDFCBANK

        # Buy Zone: Adjust support for volatility and RSI
        buy_low = support_low - (volatility * current_price * 0.1)
        buy_high = support_high if rsi > 40 else support_low

        # Earnings Outlook
        earnings_outlook = "Neutral"
        if sentiment_label == "Positive" and avg_sentiment > 0.2:
            earnings_outlook = "Positive"
        elif sentiment_label == "Negative" or (ticker == 'NXPI' and avg_sentiment < 0.1):  # NXPI Q1 2025 softness
            earnings_outlook = "Negative"
        elif ticker == 'HDFCBANK.NS' and avg_sentiment > 0.1:
            earnings_outlook = "Positive"  # Stable banking sector

        # Outlook and Detailed Rationale
        outlook = "Neutral"
        rationale = []
        if composite_score > 10 and avg_sentiment > 0.1:
            outlook = "Bullish"
            rationale.append(f"Strong Composite Score ({composite_score:.2f}) indicates robust momentum, supported by positive news sentiment ({sentiment_label}, {avg_sentiment:.2f}).")
        elif composite_score < -5 or avg_sentiment < -0.2:
            outlook = "Bearish"
            rationale.append(f"Weak Composite Score ({composite_score:.2f}) or negative news sentiment ({sentiment_label}, {avg_sentiment:.2f}) suggests caution.")
        else:
            rationale.append(f"Balanced Composite Score ({composite_score:.2f}) and {sentiment_label} sentiment ({avg_sentiment:.2f}) suggest steady performance.")
        if volatility > 0.25:
            rationale.append(f"Volatility Penalty Applied (Volatility: {volatility:.2%}), indicating higher risk.")
        if de_ratio > 2:
            rationale.append(f"High D/E Ratio ({de_ratio:.2f}) increases financial leverage risk, potentially impacting stability.")
        elif de_ratio < 1:
            rationale.append(f"Low D/E Ratio ({de_ratio:.2f}) supports financial stability, enhancing investment appeal.")
        else:
            rationale.append(f"Moderate D/E Ratio ({de_ratio:.2f}) aligns with industry norms.")
        rationale.append(f"Analyst Price Target: {'$' if ticker == 'NXPI' else '₹'}{target_price:.2f} suggests {((target_price/current_price)-1)*100:.2f}% upside.")
        rationale.append(f"Earnings Outlook: {earnings_outlook}, reflecting {sentiment_label.lower()} sentiment and recent performance trends.")
        rationale.append(f"Support Zone ({'$' if ticker == 'NXPI' else '₹'}{support_low:.2f}-{support_high:.2f}) based on 200-day SMA and recent lows; Buy Zone ({'$' if ticker == 'NXPI' else '₹'}{buy_low:.2f}-{buy_high:.2f}) adjusted for volatility and RSI ({rsi:.2f}).")

        return {
            'Ticker': ticker,
            'Support Zone Low': support_low,
            'Support Zone High': support_high,
            'Buy Range Low': buy_low,
            'Buy Range High': buy_high,
            'Target Price': target_price,
            'Earnings Outlook': earnings_outlook,
            'Outlook': outlook,
            'Rationale': "; ".join(rationale)
        }

def process_ai_predictions(input_file, output_file):
    """Process AI predictions from portfolio_rankings.xlsx"""
    try:
        df = pd.read_excel(input_file)
        predictor = AIPredictor(alpha_vantage_key='LTEQWKD1G4LWCQTM')  # Replace with your key
        ai_results = []

        for _, row in df.iterrows():
            ai_prediction = predictor.calculate_ai_predictions(row)
            ai_results.append(ai_prediction)

        # Merge AI predictions with original data
        ai_df = pd.DataFrame(ai_results)
        merged_df = df.merge(ai_df, on='Ticker', how='left')
        merged_df.to_excel(output_file, index=False)
        logging.info(f"AI predictions saved to {output_file}")
        return merged_df
    except Exception as e:
        logging.error(f"Error processing AI predictions: {e}")
        return None

def main():
    input_file = 'portfolio_rankings.xlsx'
    output_file = 'portfolio_ai_predictions.xlsx'
    process_ai_predictions(input_file, output_file)

if __name__ == "__main__":
    main()

