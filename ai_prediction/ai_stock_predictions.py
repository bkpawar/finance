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
import yfinance as yf
from newsapi import NewsApiClient

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
        """Fetch technical data using Alpha Vantage with yfinance fallback"""
        # Try Alpha Vantage first
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
            logging.error(f"Error fetching technical data for {ticker} from Alpha Vantage: {e}")
            # Fallback to yfinance
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="6mo")  # Fetch 6 months for 200-day SMA
                if len(hist) >= 200:
                    sma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
                else:
                    sma200 = hist['Close'].iloc[-1] * 0.97  # Fallback if insufficient data
                recent_low = hist['Low'].tail(30).min()
                logging.info(f"Successfully fetched technical data for {ticker} from yfinance")
                return {
                    'sma200': sma200,
                    'recent_low': recent_low
                }
            except Exception as e2:
                logging.error(f"Error fetching technical data for {ticker} from yfinance: {e2}")
                return {'sma200': 0, 'recent_low': 0}

    def fetch_sentiment(self, ticker):
        """Fetch financial news sentiment using FinBERT"""
        if not self.tokenizer or not self.model:
            self.initialize_finbert()
        try:
            # Placeholder for news headlines (replace with real news API)
            newsapi = NewsApiClient(api_key='44470d3325744bd29fb3edb45c0e40db')
            articles = newsapi.get_everything(q=ticker, language='en', sort_by='relevancy')
            headlines = [article['title'] for article in articles['articles'][:3]] or [f"Neutral news for {ticker} analysis."]
        
            headlines = [f"Neutral news for {ticker} analysis."]  # Default neutral headline
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
        """Fetch analyst price target (placeholder, replace with real API/scraping)"""
        try:
            # Placeholder: Assume 20% upside as default target
            target_price = current_price * 1.2
            return target_price
        except Exception as e:
            logging.error(f"Error fetching analyst target for {ticker}: {e}")
            return current_price * 1.2

    def calculate_ai_predictions(self, row):
        """Calculate detailed AI-driven predictions"""
        ticker = row['Ticker']
        current_price = row.get('Current Price', 0.0)
        
        # Convert numeric columns to float, handling potential string or non-numeric values
        try:
            volatility = float(row.get('Volatility', 0.2))
        except (ValueError, TypeError):
            logging.warning(f"Invalid Volatility for {ticker}. Using default 0.2.")
            volatility = 0.2
            
        try:
            rsi = float(row.get('14-Day RSI', 50.0))  # Use 14-Day RSI from input
        except (ValueError, TypeError):
            logging.warning(f"Invalid 14-Day RSI for {ticker}. Using default 50.0.")
            rsi = 50.0
            
        try:
            composite_score = float(row.get('Final Rating (0-100)', 0.0))  # Use Final Rating as Composite Score
        except (ValueError, TypeError):
            logging.warning(f"Invalid Final Rating for {ticker}. Using default 0.0.")
            composite_score = 0.0
            
        try:
            de_ratio = float(row.get('D/E Ratio', 1.0))
        except (ValueError, TypeError):
            logging.warning(f"Invalid D/E Ratio for {ticker}. Using default 1.0.")
            de_ratio = 1.0

        # Validate inputs
        if current_price <= 0:
            logging.warning(f"Invalid current price for {ticker}. Using default value.")
            current_price = 100.0  # Default fallback

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
        support_high = support_low + (5.0 if '.' not in ticker else 50.0)  # $5 for US stocks, ₹50 for Indian stocks

        # Buy Zone: Adjust support for volatility and RSI
        buy_low = support_low - (volatility * current_price * 0.1)
        buy_high = support_high if rsi > 40 else support_low

        # Earnings Outlook
        earnings_outlook = "Neutral"
        if sentiment_label == "Positive" and avg_sentiment > 0.2:
            earnings_outlook = "Positive"
        elif sentiment_label == "Negative" or avg_sentiment < -0.2:
            earnings_outlook = "Negative"

        # Outlook and Detailed Rationale
        outlook = "Neutral"
        rationale = []
        if composite_score > 60 and avg_sentiment > 0.1:
            outlook = "Bullish"
            rationale.append(f"Strong Final Rating ({composite_score:.2f}/100) indicates robust momentum, supported by positive news sentiment ({sentiment_label}, {avg_sentiment:.2f}).")
        elif composite_score < 40 or avg_sentiment < -0.2:
            outlook = "Bearish"
            rationale.append(f"Weak Final Rating ({composite_score:.2f}/100) or negative news sentiment ({sentiment_label}, {avg_sentiment:.2f}) suggests caution.")
        else:
            rationale.append(f"Moderate Final Rating ({composite_score:.2f}/100) and {sentiment_label} sentiment ({avg_sentiment:.2f}) suggest steady performance.")
        if volatility > 0.25:
            rationale.append(f"Volatility Penalty Applied (Volatility: {volatility:.2%}), indicating higher risk.")
        if de_ratio > 2:
            rationale.append(f"High D/E Ratio ({de_ratio:.2f}) increases financial leverage risk, potentially impacting stability.")
        elif de_ratio < 1:
            rationale.append(f"Low D/E Ratio ({de_ratio:.2f}) supports financial stability, enhancing investment appeal.")
        else:
            rationale.append(f"Moderate D/E Ratio ({de_ratio:.2f}) aligns with industry norms.")
        rationale.append(f"Analyst Price Target: {'$' if '.' not in ticker else '₹'}{target_price:.2f} suggests {((target_price/current_price)-1)*100:.2f}% upside.")
        rationale.append(f"Earnings Outlook: {earnings_outlook}, reflecting {sentiment_label.lower()} sentiment and recent performance trends.")
        rationale.append(f"Support Zone ({'$' if '.' not in ticker else '₹'}{support_low:.2f}-{support_high:.2f}) based on 200-day SMA and recent lows; Buy Zone ({'$' if '.' not in ticker else '₹'}{buy_low:.2f}-{buy_high:.2f}) adjusted for volatility and RSI ({rsi:.2f}).")

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

def fetch_current_price(ticker):
    """Fetch the current price of a stock using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.history(period="1d")['Close'].iloc[-1]
        logging.info(f"Fetched current price for {ticker}: {current_price}")
        return current_price
    except Exception as e:
        logging.error(f"Error fetching current price for {ticker}: {e}")
        return 0.0  # Default to 0.0 if fetching fails

def process_ai_predictions(input_file, output_file):
    """Process AI predictions from portfolio_rankings.xlsx"""
    try:
        df = pd.read_excel(input_file)
        logging.info(f"Columns in the input file: {df.columns}")
        logging.info(f"First few rows of the input file:\n{df.head()}")

        # Check for missing 'Current Price' column and fetch it if necessary
        if 'Current Price' not in df.columns:
            logging.warning("'Current Price' column is missing. Fetching prices using yfinance.")
            df['Current Price'] = df['Ticker'].apply(fetch_current_price)

        # Ensure other required columns exist
        required_columns = ['Ticker', 'Current Price', 'Volatility', '14-Day RSI', 'Final Rating (0-100)', 'D/E Ratio']
        for col in required_columns:
            if col not in df.columns:
                logging.warning(f"Missing required column: {col}. Filling with default values.")
                df[col] = 0.0  # Default value for missing columns

        # Convert numeric columns to float to avoid type issues
        for col in ['Volatility', '14-Day RSI', 'Final Rating (0-100)', 'D/E Ratio']:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                except Exception as e:
                    logging.error(f"Error converting column {col} to numeric: {e}")
                    df[col] = 0.0

        predictor = AIPredictor(alpha_vantage_key='LTEQWKD1G4LWCQTM')  # Replace with your key
        ai_results = []

        # Process all tickers before writing output
        for _, row in df.iterrows():
            try:
                ticker = row['Ticker']
                logging.info(f"Processing AI predictions for {ticker}")
                ai_prediction = predictor.calculate_ai_predictions(row)
                ai_results.append(ai_prediction)
            except Exception as e:
                logging.error(f"Error processing row for ticker {row.get('Ticker', 'Unknown')}: {e}")
                ai_results.append({
                    'Ticker': row.get('Ticker', 'Unknown'),
                    'Support Zone Low': 0.0,
                    'Support Zone High': 0.0,
                    'Buy Range Low': 0.0,
                    'Buy Range High': 0.0,
                    'Target Price': 0.0,
                    'Earnings Outlook': 'Error',
                    'Outlook': 'Error',
                    'Rationale': f'Error processing this row: {str(e)}'
                })

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
