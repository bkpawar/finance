import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from tabulate import tabulate

def prompt_user_for_inputs():
    """Prompt user for parameters that cannot be programmatically calculated."""
    print("\nPlease provide the following parameters. Instructions are provided for each.\n")

    inputs = {}

    # 1. Promoter Holding (%)
    print("Promoter Holding (%): Percentage of shares owned by promoters (founders, directors).")
    print("How to calculate:")
    print("  - Source: Check Screener.in, Moneycontrol, or NSE/BSE filings under 'Shareholding Pattern'.")
    print("  - Example: If promoters own 60% of shares, enter 60.0.")
    print("  - Default: 50.0 if unknown.")
    while True:
        try:
            value = input("Enter Promoter Holding (%) [0-100]: ") or "50.0"
            value = float(value)
            if 0 <= value <= 100:
                inputs['promoter_holding'] = value
                break
            print("Please enter a value between 0 and 100.")
        except ValueError:
            print("Please enter a valid number.")

    # 2. Institutional Holding (%)
    print("\nInstitutional Holding (%): Percentage of shares owned by institutional investors (FII + DII).")
    print("How to calculate:")
    print("  - Source: Check Screener.in or Moneycontrol under 'Shareholding Pattern' for FII and DII holdings.")
    print("  - Example: If FII = 20% and DII = 15%, enter 35.0.")
    print("  - Default: 15.0 if unknown.")
    while True:
        try:
            value = input("Enter Institutional Holding (%) [0-100]: ") or "15.0"
            value = float(value)
            if 0 <= value <= 100:
                inputs['inst_holding'] = value
                break
            print("Please enter a value between 0 and 100.")
        except ValueError:
            print("Please enter a valid number.")

    # 3. Profit Growth YoY (%)
    print("\nProfit Growth YoY (%): Year-over-year percentage growth in net profit.")
    print("How to calculate:")
    print("  - Source: Check Screener.in, Moneycontrol, or financial reports for 'Net Profit Growth' or 'PAT Growth'.")
    print("  - Formula: ((Current Year Profit - Previous Year Profit) / Previous Year Profit) * 100")
    print("  - Example: If last year's profit was ₹100 crore and this year's is ₹120 crore, enter 20.0.")
    print("  - Default: 10.0 if unknown.")
    while True:
        try:
            value = input("Enter Profit Growth YoY (%) [-100 to 100]: ") or "10.0"
            value = float(value)
            if -100 <= value <= 100:
                inputs['profit_growth'] = value
                break
            print("Please enter a value between -100 and 100.")
        except ValueError:
            print("Please enter a valid number.")

    # 4. Profit CAGR 5Y (%)
    print("\nProfit CAGR 5Y (%): 5-year Compound Annual Growth Rate of net profit.")
    print("How to calculate:")
    print("  - Source: Check Screener.in or Moneycontrol for 'Profit CAGR 5 Years'.")
    print("  - Formula: ((Ending Profit / Starting Profit)^(1/5) - 1) * 100")
    print("  - Example: If profits grew from ₹50 crore to ₹80 crore over 5 years, enter 9.86.")
    print("  - Default: 12.0 if unknown.")
    while True:
        try:
            value = input("Enter Profit CAGR 5Y (%) [-100 to 100]: ") or "12.0"
            value = float(value)
            if -100 <= value <= 100:
                inputs['cagr'] = value
                break
            print("Please enter a value between -100 and 100.")
        except ValueError:
            print("Please enter a valid number.")

    # 5. Economic Moat
    print("\nEconomic Moat: Competitive advantage of the company (None, Narrow, Wide).")
    print("How to determine:")
    print("  - Source: Analyze based on brand strength, market share, patents, or cost advantages (e.g., Morningstar reports).")
    print("  - Wide: Strong, durable advantage (e.g., Apple, Coca-Cola).")
    print("  - Narrow: Some advantage, less durable (e.g., regional leaders).")
    print("  - None: Commodity businesses or highly competitive industries.")
    print("  - Default: Narrow if unknown.")
    while True:
        value = input("Enter Economic Moat (None, Narrow, Wide): ") or "Narrow"
        value = value.capitalize()
        if value in ["None", "Narrow", "Wide"]:
            inputs['moat'] = value
            break
        print("Please enter 'None', 'Narrow', or 'Wide'.")

    # 6. Valuation Score (0-100)
    print("\nValuation Score (0-100): Subjective score of the stock's valuation attractiveness.")
    print("How to calculate:")
    print("  - Source: Compare P/E, P/B, or DCF valuation with industry averages (e.g., via Screener.in, Yahoo Finance).")
    print("  - High (70-100): Undervalued (low P/E vs. peers).")
    print("  - Medium (40-70): Fairly valued.")
    print("  - Low (0-40): Overvalued.")
    print("  - Example: If P/E is below industry average, enter 80.0.")
    print("  - Default: 50.0 if unknown.")
    while True:
        try:
            value = input("Enter Valuation Score (0-100): ") or "50.0"
            value = float(value)
            if 0 <= value <= 100:
                inputs['valuation_score'] = value
                break
            print("Please enter a value between 0 and 100.")
        except ValueError:
            print("Please enter a valid number.")

    # 7. NIFTY200DMARSIVolume (0-100)
    print("\nNIFTY200DMARSIVolume (0-100): Custom score combining performance vs. NIFTY 200, DMA, RSI, and volume.")
    print("How to calculate:")
    print("  - Source: Analyze technical performance using TradingView or other tools.")
    print("  - Factors: Outperformance vs. NIFTY 200, price above 200-day DMA, RSI 30-70, high volume.")
    print("  - Example: If stock outperforms NIFTY 200, is above DMA, RSI = 60, and volume is high, enter 80.0.")
    print("  - Default: 50.0 if unknown.")
    while True:
        try:
            value = input("Enter NIFTY200DMARSIVolume (0-100): ") or "50.0"
            value = float(value)
            if 0 <= value <= 100:
                inputs['nifty_dma_rsi_vol'] = value
                break
            print("Please enter a value between 0 and 100.")
        except ValueError:
            print("Please enter a valid number.")

    return inputs

def fetch_fundamental_data(ticker):
    """Fetch fundamental data from yfinance."""
    stock = yf.Ticker(ticker)
    info = stock.info

    data = {
        'de_ratio': 1.0,          # Default
        'roe': 15.0,              # Default
        'div_yield': 1.0,         # Default
        'symbol_trend_rs': 50.0   # Default
    }

    # Fetch available data
    if 'debtToEquity' in info and info['debtToEquity']:
        data['de_ratio'] = info['debtToEquity'] / 100
    if 'returnOnEquity' in info and info['returnOnEquity']:
        data['roe'] = info['returnOnEquity'] * 100
    if 'dividendYield' in info and info['dividendYield']:
        data['div_yield'] = info['dividendYield'] * 100

    return data

def calculate_technical_metrics(ticker, benchmark_ticker="^NSEI"):
    """Calculate technical metrics (DMA, RSI, Volume MA, RS)."""
    stock = yf.Ticker(ticker)
    benchmark = yf.Ticker(benchmark_ticker)
    stock_data = stock.history(period="1y")
    benchmark_data = benchmark.history(period="1y")

    if stock_data.empty or benchmark_data.empty:
        raise ValueError("Could not fetch historical data for stock or benchmark.")

    close = stock_data['Close']
    volume = stock_data['Volume']
    benchmark_close = benchmark_data['Close']

    # Parameters
    dma_length = 200
    rsi_length = 14
    vol_avg_length = 20
    rs_length = 63

    # 200-day DMA
    sma = SMAIndicator(close, window=dma_length).sma_indicator()
    dma_value = sma.iloc[-1] if not sma.empty else np.nan

    # 14-day RSI
    rsi = RSIIndicator(close, window=rsi_length).rsi()
    rsi_value = rsi.iloc[-1] if not rsi.empty else np.nan

    # 20-day Volume MA
    vol_avg = volume.rolling(window=vol_avg_length).mean()
    vol_avg_value = vol_avg.iloc[-1] if not vol_avg.empty else np.nan

    # Relative Strength
    if len(close) >= rs_length and len(benchmark_close) >= rs_length:
        price_change = (close.iloc[-1] - close.iloc[-rs_length]) / close.iloc[-rs_length] * 100
        benchmark_change = (benchmark_close.iloc[-1] - benchmark_close.iloc[-rs_length]) / benchmark_close.iloc[-rs_length] * 100
        rs_raw = price_change - benchmark_change
        rs_scaled = max(0, min(100, 50 + rs_raw))
    else:
        rs_scaled = np.nan

    return {
        'dma_value': dma_value,
        'rsi_value': rsi_value,
        'vol_avg': vol_avg_value,
        'rs_scaled': rs_scaled,
        'current_price': close.iloc[-1],
        'current_volume': volume.iloc[-1]
    }

def calculate_final_rating(fundamental_data, user_inputs, technical_data):
    """Calculate scores for all parameters and compute final rating."""
    # Combine fundamental and user inputs
    data = {**fundamental_data, **user_inputs}

    # Scores
    score_prom_holding = min(100, data['promoter_holding'] * 2)
    score_inst_holding = min(100, data['inst_holding'] * 4)
    score_de_ratio = max(0, 100 - data['de_ratio'] * 50)
    score_roe = min(100, data['roe'] * 5)
    score_profit_growth = min(100, data['profit_growth'] * 5)
    score_cagr = min(100, data['cagr'] * 5)
    score_div_yield = min(100, data['div_yield'] * 50)
    score_moat = 100 if data['moat'] == 'Wide' else 50 if data['moat'] == 'Narrow' else 0
    score_valuation = data['valuation_score']
    score_symbol_trend_rs = data['symbol_trend_rs']
    score_nifty_dma_rsi_vol = data['nifty_dma_rsi_vol']
    score_dma = 100 if technical_data['current_price'] > technical_data['dma_value'] else 0
    score_rsi = 25 if technical_data['rsi_value'] < 30 else 75 if technical_data['rsi_value'] > 70 else 50
    score_vol_avg = 75 if technical_data['current_volume'] > technical_data['vol_avg'] else 25
    score_rs = 50 if np.isnan(technical_data['rs_scaled']) else technical_data['rs_scaled']

    # Sum scores and calculate final rating
    scores = [
        score_prom_holding, score_inst_holding, score_de_ratio, score_roe, score_profit_growth,
        score_cagr, score_div_yield, score_moat, score_valuation, score_symbol_trend_rs,
        score_nifty_dma_rsi_vol, score_dma, score_rsi, score_vol_avg, score_rs
    ]
    final_rating = round(sum(scores) / len(scores), 2)

    return scores, final_rating

def display_table(fundamental_data, user_inputs, technical_data, scores, final_rating):
    """Display the results in a table format."""
    data = {**fundamental_data, **user_inputs}
    table_data = [
        ["Parameter", "Value"],
        ["SymbolTrendRS (Manual)", f"{data['symbol_trend_rs']:.2f}"],
        ["NIFTY200DMARSIVolume (Manual)", f"{data['nifty_dma_rsi_vol']:.2f}"],
        ["Promoter Holding (%)", f"{data['promoter_holding']:.2f}"],
        ["Inst. Holding (%)", f"{data['inst_holding']:.2f}"],
        ["D/E Ratio", f"{data['de_ratio']:.2f}"],
        ["RoE (%)", f"{data['roe']:.2f}"],
        ["Profit Growth YoY (%)", f"{data['profit_growth']:.2f}"],
        ["Dividend Yield (%)", f"{data['div_yield']:.2f}"],
        ["Economic Moat", data['moat']],
        ["Profit CAGR 5Y (%)", f"{data['cagr']:.2f}"],
        ["Valuation Score", f"{data['valuation_score']:.2f}"],
        ["Calculated Metrics", ""],
        [f"200-Day DMA", f"{technical_data['dma_value']:.2f}"],
        [f"14-Day RSI", f"{technical_data['rsi_value']:.2f}"],
        [f"20-Day Vol MA", f"{technical_data['vol_avg']:.2f}"],
        [f"Simple RS vs NIFTY_50 (63d)", "Invalid Symbol" if np.isnan(technical_data['rs_scaled']) else f"{technical_data['rs_scaled']:.2f}"],
        ["Final Rating (0-100)", f"{final_rating:.2f}"]
    ]

    print("\nStock Info Table")
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

def main():
    # Prompt user for NSE ticker
    ticker = input("Enter the NSE stock ticker (e.g., RELIANCE.NS): ").strip().upper()
    if not ticker.endswith('.NS'):
        ticker += '.NS'

    try:
        # Fetch user inputs for non-programmable parameters
        user_inputs = prompt_user_for_inputs()

        # Fetch fundamental data
        fundamental_data = fetch_fundamental_data(ticker)

        # Calculate technical metrics
        technical_data = calculate_technical_metrics(ticker, benchmark_ticker="^NSEI")

        # Calculate scores and final rating
        scores, final_rating = calculate_final_rating(fundamental_data, user_inputs, technical_data)

        # Display results
        display_table(fundamental_data, user_inputs, technical_data, scores, final_rating)

    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure the ticker is valid and try again.")

if __name__ == "__main__":
    main()
