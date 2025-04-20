Below is the content for a `README.md` file for your AI stock prediction tool (`ai_stock_predictions.py`). The README provides an overview of the tool, its functionality, prerequisites, installation steps, usage instructions, and troubleshooting tips. It is tailored to the script's features, including dynamic ticker processing from an Excel file, technical data fetching with Alpha Vantage and `yfinance` fallback, and placeholder logic for sentiment and analyst targets.

---

# AI Stock Prediction Tool

## Overview

The **AI Stock Prediction Tool** is a Python-based application that analyzes stocks listed in an input Excel file (`portfolio_rankings.xlsx`) and generates AI-driven investment predictions. The tool processes fundamental and technical data to provide insights such as support and buy zones, target prices, earnings outlook, and investment rationale. It is designed to handle any number of tickers dynamically, supporting both US (e.g., `NXPI`) and non-US (e.g., `HDFCBANK.NS`) stocks.

### Features
- **Dynamic Ticker Processing**: Reads tickers from an input Excel file and processes them without hardcoded stock names.
- **Technical Analysis**: Fetches 200-day SMA and 30-day low using Alpha Vantage, with a fallback to `yfinance` for non-US tickers.
- **Fundamental Analysis**: Uses input data (e.g., `Final Rating (0-100)`, `14-Day RSI`, `D/E Ratio`) to assess stock performance.
- **Sentiment Analysis**: Applies a neutral sentiment placeholder (extendable with a news API like NewsAPI).
- **Analyst Targets**: Uses a 20% upside placeholder (extendable with `yfinance` or other APIs).
- **Output**: Generates an Excel file (`portfolio_ai_predictions.xlsx`) with predictions merged with input data.
- **Error Handling**: Robust handling for missing data, type mismatches, and API failures.

## Prerequisites

- **Python**: Version 3.6 or higher.
- **Dependencies**:
  - `pandas`: For Excel file processing.
  - `numpy`: For numerical computations.
  - `requests`: For API calls.
  - `beautifulsoup4`: For potential web scraping (not currently used).
  - `alpha_vantage`: For technical data.
  - `yfinance`: For price and technical data fallback.
  - `transformers`: For FinBERT sentiment analysis.
  - `torch`: For FinBERT model inference.
- **Alpha Vantage API Key**: Obtain a free key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key).
- **Input Excel File**: A `portfolio_rankings.xlsx` file with at least a `Ticker` column. Recommended columns:
  - `Final Rating (0-100)`: Numeric score (0–100).
  - `14-Day RSI`: Relative Strength Index (0–100).
  - `D/E Ratio`: Debt-to-Equity ratio.
  - `Volatility`: Stock volatility (e.g., 0.20 for 20%).
  - If missing, defaults are applied (e.g., `Volatility=0.2`, `14-Day RSI=50.0`).

## Installation

1. **Clone or Download the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install pandas numpy requests beautifulsoup4 alpha_vantage yfinance transformers torch
   ```

4. **Configure Alpha Vantage API Key**:
   - Replace `'LTEQWKD1G4LWCQTM'` in `ai_stock_predictions.py` with your Alpha Vantage API key:
     ```python
     predictor = AIPredictor(alpha_vantage_key='YOUR_API_KEY')
     ```

## Usage

1. **Prepare the Input Excel File**:
   - Create `portfolio_rankings.xlsx` in the project directory.
   - Ensure it has a `Ticker` column (e.g., `NXPI`, `HDFCBANK.NS`).
   - Optionally include `Final Rating (0-100)`, `14-Day RSI`, `D/E Ratio`, and `Volatility` for better predictions.
   - Example structure:
     ```excel
     Ticker       | Final Rating (0-100) | 14-Day RSI | D/E Ratio | Volatility
     NXPI         | 33.81                | 50         | 116.945   | 0.50
     HDFCBANK.NS  | 66.47                | 75         | 0.85      | 0.20
     ```

2. **Run the Script**:
   ```bash
   python3 ai_stock_predictions.py
   ```

3. **Check the Output**:
   - The script generates `portfolio_ai_predictions.xlsx` in the project directory.
   - The output includes input columns plus:
     - `Support Zone Low/High`: Predicted price support range.
     - `Buy Range Low/High`: Suggested buying range.
     - `Target Price`: Analyst target (placeholder: 20% upside).
     - `Earnings Outlook`: Positive, Neutral, or Negative.
     - `Outlook`: Bullish, Neutral, or Bearish.
     - `Rationale`: Detailed explanation of predictions.

## Output Example

For input tickers `NXPI` and `HDFCBANK.NS`, the output Excel file might include:

```excel
Ticker       | Final Rating (0-100) | 14-Day RSI | D/E Ratio | Current Price | Support Zone Low | Support Zone High | Buy Range Low | Buy Range High | Target Price | Earnings Outlook | Outlook | Rationale
NXPI         | 33.81                | 50         | 116.945   | 170.74        | 165.62           | 170.62            | 163.90        | 165.62         | 204.89       | Neutral         | Bearish | Weak Final Rating (33.81/100) suggests caution; High D/E Ratio (116.95) increases risk...
HDFCBANK.NS  | 66.47                | 75         | 0.85      | 1906.70       | 1850.30          | 1900.30           | 1831.63       | 1900.30        | 2288.04      | Neutral         | Bullish | Strong Final Rating (66.47/100) indicates robust momentum; Low D/E Ratio (0.85) supports stability...
```

## Troubleshooting

1. **Error: `Error fetching technical data for <ticker>: Error getting data from the api`**
   - **Cause**: Alpha Vantage may not support non-US tickers (e.g., `HDFCBANK.NS`) or the API key is invalid/rate-limited.
   - **Solution**:
     - The script uses `yfinance` as a fallback, which should handle `HDFCBANK.NS`.
     - Verify your Alpha Vantage API key.
     - Check rate limits (free tier: 5 calls/min). Wait or upgrade to a premium plan.
     - Ensure internet connectivity.

2. **Error: `'">' not supported between instances of 'str' and 'int'`**
   - **Cause**: Numeric columns (e.g., `14-Day RSI`) are strings in the input Excel file.
   - **Solution**: The script converts these to floats. Verify that `portfolio_rankings.xlsx` has numeric values (e.g., `50` instead of `"50"`).

3. **Warning: `Missing required column: Volatility`**
   - **Cause**: The input Excel file lacks a `Volatility` column.
   - **Solution**: The script uses a default (`0.2`). Add a `Volatility` column to the Excel file for accurate predictions (e.g., `0.50` for `NXPI`).

4. **LibreSSL Warning**:
   - **Cause**: `urllib3` is compiled with LibreSSL instead of OpenSSL 1.1.1+.
   - **Solution**:
     - Update `urllib3`: `pip install --upgrade urllib3`.
     - Install OpenSSL 1.1.1+ or use a Python environment with OpenSSL support.
     - This warning does not affect functionality.

5. **No Predictions in Output**:
   - **Cause**: API failures or invalid input data.
   - **Solution**:
     - Check logs in the terminal for errors.
     - Ensure `portfolio_rankings.xlsx` has a valid `Ticker` column.
     - Verify API key and internet connectivity.

## Extending the Tool

1. **Add Real Sentiment Analysis**:
   - Integrate NewsAPI for dynamic news headlines:
     ```python
     from newsapi import NewsApiClient
     def fetch_sentiment(self, ticker):
         newsapi = NewsApiClient(api_key='YOUR_NEWSAPI_KEY')
         articles = newsapi.get_everything(q=ticker, language='en', sort_by='relevancy')
         headlines = [article['title'] for article in articles['articles'][:3]] or [f"Neutral news for {ticker} analysis."]
         # Proceed with FinBERT analysis
     ```
   - Install NewsAPI: `pip install newsapi-python`.
   - Obtain a key from [NewsAPI](https://newsapi.org/).

2. **Add Real Analyst Targets**:
   - Use `yfinance` for analyst price targets:
     ```python
     def fetch_analyst_target(self, ticker, current_price):
         stock = yf.Ticker(ticker)
         target = stock.info.get('targetMeanPrice', current_price * 1.2)
         return target
     ```

3. **Alternative Data Providers**:
   - If Alpha Vantage fails for non-US tickers, rely solely on `yfinance` or use other providers like Tiingo or IEX Cloud.

## Limitations

- **Alpha Vantage**: Limited support for non-US tickers (e.g., `HDFCBANK.NS`). The `yfinance` fallback mitigates this.
- **Placeholders**: Sentiment and analyst targets use placeholders (neutral sentiment, 20% upside). Integrate real APIs for better accuracy.
- **Rate Limits**: Alpha Vantage free tier allows 5 calls/min. Multiple tickers may require waiting or a premium plan.
- **Input Dependency**: Predictions rely on the quality of input data (e.g., `Final Rating`, `RSI`). Missing or incorrect data may affect results.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit changes: `git commit -m "Add feature"`.
4. Push to the branch: `git push origin feature-name`.
5. Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions or support, contact [bkpawar@gmail.com] or open an issue on the repository.
