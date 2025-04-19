import logging
import pandas as pd
import numpy as np
from yahooquery import Ticker
import yfinance as yf
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
import configparser
from tabulate import tabulate
from prompt_toolkit import PromptSession
import matplotlib.pyplot as plt
import os

# Setup logging for debugging and error tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration from config.ini
config = configparser.ConfigParser()
config['DEFAULT'] = {
    'benchmark_ticker': '^NSEI',  # NIFTY 50 index
    'dma_length': '200',         # Days for DMA calculation
    'rsi_length': '14',          # Days for RSI
    'vol_avg_length': '20',      # Days for volume MA
    'rs_length': '63',           # Days for Relative Strength
    'weights': '0.1,0.1,0.1,0.1,0.1,0.1,0.05,0.05,0.15,0.1,0.1,0.05,0.05,0.05,0.1'  # Weights for 15 parameters
}
if os.path.exists('config.ini'):
    config.read('config.ini')
else:
    with open('config.ini', 'w') as f:
        config.write(f)

class StockAnalyzer:
    """
    Class to analyze a single NSE stock and compute its Final Rating.
    Handles data fetching, calculations, user inputs, and output generation.
    """
    def __init__(self, ticker):
        """Initialize with stock ticker and load configuration."""
        self.ticker = ticker
        self.benchmark_ticker = config['DEFAULT']['benchmark_ticker']
        self.params = {
            'dma_length': int(config['DEFAULT']['dma_length']),
            'rsi_length': int(config['DEFAULT']['rsi_length']),
            'vol_avg_length': int(config['DEFAULT']['vol_avg_length']),
            'rs_length': int(config['DEFAULT']['rs_length'])
        }
        self.weights = [float(w) for w in config['DEFAULT']['weights'].split(',')]
        self.fundamental_data = {}  # Store fundamental metrics
        self.technical_data = {}   # Store technical metrics
        self.user_inputs = {}      # Store user-provided inputs
        self.moat_basis = ""       # Store Economic Moat calculation details
        self.calculation_details = []  # Store details for auto-calculated parameters

    def fetch_fundamental_data(self):
        """
        Fetch fundamental data using yahooquery (primary) and yfinance (fallback).
        Calculates Economic Moat and Valuation Score programmatically.
        """
        try:
            stock = Ticker(self.ticker)
            info = stock.summary_detail.get(self.ticker, {})
            yf_stock = yf.Ticker(self.ticker)

            # Initialize default values
            self.fundamental_data = {
                'promoter_holding': 50.0,  # User input
                'inst_holding': 15.0,      # User input
                'de_ratio': 1.0,
                'roe': 15.0,
                'profit_growth': 10.0,     # User input
                'cagr': 12.0,              # User input
                'div_yield': 1.0,
                'moat': 'Narrow',
                'valuation_score': 50.0,
                'symbol_trend_rs': 50.0,
                'nifty_dma_rsi_vol': 50.0
            }

            # Fetch from yahooquery
            if 'debtToEquity' in stock.financial_data.get(self.ticker, {}):
                self.fundamental_data['de_ratio'] = stock.financial_data[self.ticker]['debtToEquity'] / 100
                self.calculation_details.append(
                    f"D/E Ratio: {self.fundamental_data['de_ratio']:.2f} (Fetched from yahooquery)"
                )
            if 'returnOnEquity' in stock.financial_data.get(self.ticker, {}):
                self.fundamental_data['roe'] = stock.financial_data[self.ticker]['returnOnEquity'] * 100
                self.calculation_details.append(
                    f"RoE: {self.fundamental_data['roe']:.2f}% (Fetched from yahooquery)"
                )
            if 'dividendYield' in info:
                self.fundamental_data['div_yield'] = info['dividendYield'] * 100
                self.calculation_details.append(
                    f"Dividend Yield: {self.fundamental_data['div_yield']:.2f}% (Fetched from yahooquery)"
                )

            # Valuation Score based on P/E and P/B
            pe_ratio = stock.key_stats.get(self.ticker, {}).get('trailingPE', np.nan)
            pb_ratio = stock.key_stats.get(self.ticker, {}).get('priceToBook', np.nan)
            industry_pe = 30.0  # Placeholder; ideally dynamic
            if not np.isnan(pe_ratio) and not np.isnan(pb_ratio):
                pe_score = max(0, min(100, 100 - (pe_ratio / industry_pe) * 50))
                pb_score = max(0, min(100, 100 - (pb_ratio / 5) * 50))
                self.fundamental_data['valuation_score'] = (pe_score + pb_score) / 2
                self.calculation_details.append(
                    f"Valuation Score: {self.fundamental_data['valuation_score']:.2f} "
                    f"(Based on P/E={pe_ratio:.2f} vs industry P/E={industry_pe}, "
                    f"P/B={pb_ratio:.2f})"
                )

            # Fallback to yfinance
            yf_info = yf_stock.info
            if np.isnan(self.fundamental_data['de_ratio']) and 'debtToEquity' in yf_info and yf_info['debtToEquity']:
                self.fundamental_data['de_ratio'] = yf_info['debtToEquity'] / 100
                self.calculation_details.append(
                    f"D/E Ratio: {self.fundamental_data['de_ratio']:.2f} (Fetched from yfinance)"
                )
            if np.isnan(self.fundamental_data['roe']) and 'returnOnEquity' in yf_info and yf_info['returnOnEquity']:
                self.fundamental_data['roe'] = yf_info['returnOnEquity'] * 100
                self.calculation_details.append(
                    f"RoE: {self.fundamental_data['roe']:.2f}% (Fetched from yfinance)"
                )
            if np.isnan(self.fundamental_data['div_yield']) and 'dividendYield' in yf_info and yf_info['dividendYield']:
                self.fundamental_data['div_yield'] = yf_info['dividendYield'] * 100
                self.calculation_details.append(
                    f"Dividend Yield: {self.fundamental_data['div_yield']:.2f}% (Fetched from yfinance)"
                )

            # Economic Moat Estimation
            profit_margin = stock.financial_data.get(self.ticker, {}).get('profitMargins', 0)
            market_cap = stock.summary_detail.get(self.ticker, {}).get('marketCap', 0)
            roic = stock.financial_data.get(self.ticker, {}).get('returnOnCapitalEmployed', 0)
            revenue_stability = 0.1  # Placeholder; ideally from historical revenue
            moat_factors = []

            if profit_margin > 0.2:
                moat_factors.append(f"High profit margin ({profit_margin:.2%})")
            if market_cap > 1e12:
                moat_factors.append(f"Large market cap ({market_cap/1e9:.2f}B INR)")
            if roic > 0.15:
                moat_factors.append(f"High ROIC ({roic:.2%})")
            if revenue_stability < 0.05:
                moat_factors.append(f"Stable revenue (std {revenue_stability:.2%})")

            if len(moat_factors) >= 3:
                self.fundamental_data['moat'] = 'Wide'
                self.moat_basis = "Wide moat due to: " + ", ".join(moat_factors)
            elif len(moat_factors) >= 1:
                self.fundamental_data['moat'] = 'Narrow'
                self.moat_basis = "Narrow moat due to: " + ", ".join(moat_factors)
            else:
                self.fundamental_data['moat'] = 'None'
                self.moat_basis = "No moat: Insufficient competitive advantage"

            self.calculation_details.append(
                f"Economic Moat: {self.fundamental_data['moat']} ({self.moat_basis}; "
                f"Profit Margin={profit_margin:.2%}, Market Cap={market_cap/1e9:.2f}B INR, "
                f"ROIC={roic:.2%}, Revenue Stability={revenue_stability:.2%})"
            )

            logger.info(f"Fetched fundamental data for {self.ticker}")
        except Exception as e:
            logger.error(f"Failed to fetch fundamental data: {e}")
            raise

    def prompt_user_for_inputs(self):
        """
        Prompt user for parameters that cannot be automated (4 inputs).
        Allow override of calculated Economic Moat.
        """
        session = PromptSession(multiline=False)
        print("\nPlease provide the following parameters. Instructions are provided.\n")

        # Promoter Holding
        print("Promoter Holding (%): Percentage of shares owned by promoters.")
        print("  - Source: Screener.in, Moneycontrol, or NSE/BSE under 'Shareholding Pattern'.")
        print("  - Example: If promoters own 60%, enter 60.0. Default: 50.0")
        while True:
            value = session.prompt("Enter Promoter Holding (%) [0-100]: ", default="50.0")
            try:
                value = float(value)
                if 0 <= value <= 100:
                    self.user_inputs['promoter_holding'] = value
                    break
                print("Please enter a value between 0 and 100.")
            except ValueError:
                print("Please enter a valid number.")

        # Institutional Holding
        print("\nInstitutional Holding (%): Percentage owned by FII + DII.")
        print("  - Source: Screener.in or Moneycontrol under 'Shareholding Pattern'.")
        print("  - Example: FII = 20%, DII = 15%, enter 35.0. Default: 15.0")
        while True:
            value = session.prompt("Enter Institutional Holding (%) [0-100]: ", default="15.0")
            try:
                value = float(value)
                if 0 <= value <= 100:
                    self.user_inputs['inst_holding'] = value
                    break
                print("Please enter a value between 0 and 100.")
            except ValueError:
                print("Please enter a valid number.")

        # Profit Growth YoY
        print("\nProfit Growth YoY (%): Year-over-year net profit growth.")
        print("  - Source: Screener.in or financial reports for 'Net Profit Growth'.")
        print("  - Formula: ((Current Profit - Previous Profit) / Previous Profit) * 100")
        print("  - Example: Current = ₹120 crore, Previous = ₹100 crore, enter 20.0. Default: 10.0")
        while True:
            value = session.prompt("Enter Profit Growth YoY (%) [-100 to 100]: ", default="10.0")
            try:
                value = float(value)
                if -100 <= value <= 100:
                    self.user_inputs['profit_growth'] = value
                    break
                print("Please enter a value between -100 and 100.")
            except ValueError:
                print("Please enter a valid number.")

        # Profit CAGR 5Y
        print("\nProfit CAGR 5Y (%): 5-year Compound Annual Growth Rate of net profit.")
        print("  - Source: Screener.in for 'Profit CAGR 5 Years'.")
        print("  - Formula: ((Ending Profit / Starting Profit)^(1/5) - 1) * 100")
        print("  - Example: From ₹50 crore to ₹80 crore over 5 years, enter 9.86. Default: 12.0")
        while True:
            value = session.prompt("Enter Profit CAGR 5Y (%) [-100 to 100]: ", default="12.0")
            try:
                value = float(value)
                if -100 <= value <= 100:
                    self.user_inputs['cagr'] = value
                    break
                print("Please enter a value between -100 and 100.")
            except ValueError:
                print("Please enter a valid number.")

        # Economic Moat Override
        print(f"\nEconomic Moat: Calculated as '{self.fundamental_data['moat']}'.")
        print(f"  - Basis: {self.moat_basis}")
        print("  - Override if you disagree (e.g., based on Morningstar or industry analysis).")
        print("  - Options: None, Narrow, Wide. Default: Accept calculated value.")
        override = session.prompt("Override Economic Moat? (Enter None, Narrow, Wide, or press Enter to accept): ")
        if override.capitalize() in ['None', 'Narrow', 'Wide']:
            self.fundamental_data['moat'] = override.capitalize()
            self.moat_basis += f" (Overridden by user to {self.fundamental_data['moat']})"
            self.calculation_details[-1] = (
                f"Economic Moat: {self.fundamental_data['moat']} ({self.moat_basis})"
            )

    def calculate_technical_metrics(self):
        """
        Calculate technical metrics (DMA, RSI, Volume MA, RS, SymbolTrendRS, NIFTY200DMARSIVolume).
        Includes volatility for risk adjustment.
        """
        try:
            stock = yf.Ticker(self.ticker)
            benchmark = yf.Ticker(self.benchmark_ticker)
            stock_data = stock.history(period="1y")
            benchmark_data = benchmark.history(period="1y")

            if stock_data.empty or benchmark_data.empty:
                raise ValueError("Could not fetch historical data.")

            close = stock_data['Close']
            volume = stock_data['Volume']
            benchmark_close = benchmark_data['Close']

            # Technical indicators
            sma = SMAIndicator(close, window=self.params['dma_length']).sma_indicator()
            rsi = RSIIndicator(close, window=self.params['rsi_length']).rsi()
            vol_avg = volume.rolling(window=self.params['vol_avg_length']).mean()
            macd = MACD(close).macd_signal()

            self.technical_data = {
                'dma_value': sma.iloc[-1] if not sma.empty else np.nan,
                'rsi_value': rsi.iloc[-1] if not rsi.empty else np.nan,
                'vol_avg': vol_avg.iloc[-1] if not vol_avg.empty else np.nan,
                'current_price': close.iloc[-1],
                'current_volume': volume.iloc[-1],
                'macd_signal': macd.iloc[-1] if not macd.empty else 0
            }

            # Relative Strength
            if len(close) >= self.params['rs_length'] and len(benchmark_close) >= self.params['rs_length']:
                price_change = (close.iloc[-1] - close.iloc[-self.params['rs_length']]) / close.iloc[-self.params['rs_length']] * 100
                benchmark_change = (benchmark_close.iloc[-1] - benchmark_close.iloc[-self.params['rs_length']]) / benchmark_close.iloc[-self.params['rs_length']] * 100
                rs_raw = price_change - benchmark_change
                self.technical_data['rs_scaled'] = max(0, min(100, 50 + rs_raw))
            else:
                self.technical_data['rs_scaled'] = np.nan

            self.calculation_details.append(
                f"{self.params['dma_length']}-Day DMA: {self.technical_data['dma_value']:.2f} "
                f"(Simple Moving Average of closing prices over {self.params['dma_length']} days)"
            )
            self.calculation_details.append(
                f"{self.params['rsi_length']}-Day RSI: {self.technical_data['rsi_value']:.2f} "
                f"(Relative Strength Index over {self.params['rsi_length']} days)"
            )
            self.calculation_details.append(
                f"{self.params['vol_avg_length']}-Day Vol MA: {self.technical_data['vol_avg']:.2f} "
                f"(Average trading volume over {self.params['vol_avg_length']} days)"
            )
            rs_display = "Invalid Symbol" if np.isnan(self.technical_data['rs_scaled']) else f"{self.technical_data['rs_scaled']:.2f}"
            self.calculation_details.append(
                f"Simple RS vs NIFTY_50 ({self.params['rs_length']}d): {rs_display} "
                f"(Stock's {self.params['rs_length']}-day return vs NIFTY 50, scaled 0-100)"
            )

            # SymbolTrendRS
            rsi_score = (self.technical_data['rsi_value'] / 100) * 50
            macd_score = 50 if self.technical_data['macd_signal'] > 0 else 25
            self.fundamental_data['symbol_trend_rs'] = (rsi_score + macd_score) / 2
            self.calculation_details.append(
                f"SymbolTrendRS: {self.fundamental_data['symbol_trend_rs']:.2f} "
                f"(Average of RSI-based score ({rsi_score:.2f}) and MACD signal score ({macd_score:.2f}))"
            )

            # NIFTY200DMARSIVolume
            dma_score = 25 if self.technical_data['current_price'] > self.technical_data['dma_value'] else 0
            rsi_score = 25 if 30 <= self.technical_data['rsi_value'] <= 70 else 10
            vol_score = 25 if self.technical_data['current_volume'] > self.technical_data['vol_avg'] else 10
            rs_score = 25 if not np.isnan(self.technical_data['rs_scaled']) and self.technical_data['rs_scaled'] > 50 else 10
            self.fundamental_data['nifty_dma_rsi_vol'] = dma_score + rsi_score + vol_score + rs_score
            self.calculation_details.append(
                f"NIFTY200DMARSIVolume: {self.fundamental_data['nifty_dma_rsi_vol']:.2f} "
                f"(Composite score: DMA={dma_score}, RSI={rsi_score}, Volume={vol_score}, RS={rs_score})"
            )

            # Volatility
            volatility = close.pct_change().std() * np.sqrt(252)
            self.technical_data['volatility'] = volatility
            self.calculation_details.append(
                f"Volatility: {self.technical_data['volatility']:.2%} "
                f"(Annualized standard deviation of daily returns)"
            )

            logger.info(f"Calculated technical metrics for {self.ticker}")
        except Exception as e:
            logger.error(f"Failed to calculate technical metrics: {e}")
            raise

    def calculate_final_rating(self):
        """
        Compute scores for all parameters and calculate weighted Final Rating.
        Applies volatility penalty for risk adjustment.
        """
        data = {**self.fundamental_data, **self.user_inputs}

        scores = [
            min(100, data['promoter_holding'] * 2),  # Promoter Holding
            min(100, data['inst_holding'] * 4),      # Institutional Holding
            max(0, 100 - data['de_ratio'] * 50),     # D/E Ratio
            min(100, data['roe'] * 5),               # RoE
            min(100, data['profit_growth'] * 5),     # Profit Growth YoY
            min(100, data['cagr'] * 5),              # Profit CAGR 5Y
            min(100, data['div_yield'] * 50),        # Dividend Yield
            100 if data['moat'] == 'Wide' else 50 if data['moat'] == 'Narrow' else 0,  # Moat
            data['valuation_score'],                  # Valuation Score
            data['symbol_trend_rs'],                  # SymbolTrendRS
            data['nifty_dma_rsi_vol'],                # NIFTY200DMARSIVolume
            100 if self.technical_data['current_price'] > self.technical_data['dma_value'] else 0,  # DMA
            25 if self.technical_data['rsi_value'] < 30 else 75 if self.technical_data['rsi_value'] > 70 else 50,  # RSI
            75 if self.technical_data['current_volume'] > self.technical_data['vol_avg'] else 25,  # Volume
            50 if np.isnan(self.technical_data['rs_scaled']) else self.technical_data['rs_scaled']  # RS
        ]

        volatility_penalty = max(0, 100 - (self.technical_data['volatility'] * 100)) / 100
        weighted_scores = [score * weight * volatility_penalty for score, weight in zip(scores, self.weights)]
        final_rating = round(sum(weighted_scores) / sum(self.weights), 2)

        self.calculation_details.append(
            f"Final Rating: {final_rating:.2f} "
            f"(Weighted average of 15 scores with volatility penalty {volatility_penalty:.2%})"
        )

        return scores, final_rating

    def display_table(self, scores, final_rating):
        """
        Display results in a table and list calculation details for auto-calculated parameters.
        Save results to CSV.
        """
        data = {**self.fundamental_data, **self.user_inputs}
        table_data = [
            ["Parameter", "Value"],
            ["SymbolTrendRS", f"{data['symbol_trend_rs']:.2f}"],
            ["NIFTY200DMARSIVolume", f"{data['nifty_dma_rsi_vol']:.2f}"],
            ["Promoter Holding (%)", f"{data['promoter_holding']:.2f}"],
            ["Inst. Holding (%)", f"{data['inst_holding']:.2f}"],
            ["D/E Ratio", f"{data['de_ratio']:.2f}"],
            ["RoE (%)", f"{data['roe']:.2f}"],
            ["Profit Growth YoY (%)", f"{data['profit_growth']:.2f}"],
            ["Dividend Yield (%)", f"{data['div_yield']:.2f}"],
            ["Economic Moat", f"{data['moat']} ({self.moat_basis})"],
            ["Profit CAGR 5Y (%)", f"{data['cagr']:.2f}"],
            ["Valuation Score", f"{data['valuation_score']:.2f}"],
            ["Calculated Metrics", ""],
            [f"{self.params['dma_length']}-Day DMA", f"{self.technical_data['dma_value']:.2f}"],
            [f"{self.params['rsi_length']}-Day RSI", f"{self.technical_data['rsi_value']:.2f}"],
            [f"{self.params['vol_avg_length']}-Day Vol MA", f"{self.technical_data['vol_avg']:.2f}"],
            [f"Simple RS vs NIFTY_50 ({self.params['rs_length']}d)", "Invalid Symbol" if np.isnan(self.technical_data['rs_scaled']) else f"{self.technical_data['rs_scaled']:.2f}"],
            ["Final Rating (0-100)", f"{final_rating:.2f}"]
        ]

        print("\nStock Info Table")
        print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

        print("\nCalculation Details for Automatically Calculated Parameters:")
        for detail in self.calculation_details:
            print(f"- {detail}")

        df = pd.DataFrame(table_data[1:], columns=table_data[0])
        df.to_csv(f"{self.ticker}_analysis.csv", index=False)
        logger.info(f"Saved results to {self.ticker}_analysis.csv")

    def plot_metrics(self):
        """Generate and save a plot of stock price vs. 200-Day DMA."""
        stock = yf.Ticker(self.ticker)
        data = stock.history(period="1y")
        sma = SMAIndicator(data['Close'], window=self.params['dma_length']).sma_indicator()

        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data['Close'], label='Close Price')
        plt.plot(data.index, sma, label=f"{self.params['dma_length']}-Day DMA")
        plt.title(f"{self.ticker} Price and DMA")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.savefig(f"{self.ticker}_plot.png")
        plt.close()
        logger.info(f"Saved plot to {self.ticker}_plot.png")

def main():
    """Main function to run the stock analysis."""
    session = PromptSession(multiline=False)
    ticker = session.prompt("Enter the NSE stock ticker (e.g., RELIANCE.NS): ").strip().upper()
    if not ticker.endswith('.NS'):
        ticker += '.NS'

    try:
        analyzer = StockAnalyzer(ticker)
        analyzer.fetch_fundamental_data()
        analyzer.prompt_user_for_inputs()
        analyzer.calculate_technical_metrics()
        scores, final_rating = analyzer.calculate_final_rating()
        analyzer.display_table(scores, final_rating)
        analyzer.plot_metrics()
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print("Please ensure the ticker is valid and try again.")

if __name__ == "__main__":
    main()

