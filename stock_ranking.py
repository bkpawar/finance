import configparser
import os
import pandas as pd
import numpy as np
from yahooquery import Ticker
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD
import matplotlib.pyplot as plt
from tabulate import tabulate
import argparse
from prompt_toolkit import PromptSession
import logging
import time

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Get list of supported exchanges from config.ini
SUPPORTED_EXCHANGES = [section for section in config.sections() if section != 'DEFAULT']

class StockAnalyzer:
    """
    Class to analyze a single stock from any exchange and compute its Final Rating.
    Handles data fetching, calculations, user inputs, and output generation.
    """
    def __init__(self, ticker, exchange='NSE'):
        """Initialize with stock ticker, exchange, and load configuration."""
        self.ticker = ticker
        self.exchange = exchange.upper()
        
        # Validate exchange
        if self.exchange not in SUPPORTED_EXCHANGES:
            raise ValueError(f"Unsupported exchange: {self.exchange}. Supported exchanges: {', '.join(SUPPORTED_EXCHANGES)}")
        
        # Load exchange-specific settings from config.ini
        self.suffix = config[self.exchange].get('suffix', '')
        self.benchmark_ticker = config[self.exchange].get('benchmark_ticker')
        
        self.params = {
            'dma_length': int(config['DEFAULT']['dma_length']),
            'rsi_length': int(config['DEFAULT']['rsi_length']),
            'vol_avg_length': int(config['DEFAULT']['vol_avg_length']),
            'rs_length': int(config['DEFAULT']['rs_length'])
        }
        self.weights = [float(w) for w in config['DEFAULT']['weights'].split(',')]
        self.fundamental_data = {}
        self.technical_data = {}
        self.user_inputs = {}
        self.moat_basis = ""
        self.calculation_details = []
        self.final_rating = None

    def fetch_fundamental_data(self):
        """Fetch fundamental data using yahooquery and yfinance as fallback."""
        try:
            stock = Ticker(self.ticker)
            self.fundamental_data = stock.financial_data.get(self.ticker, {})
            if not self.fundamental_data:
                logger.warning(f"No fundamental data from yahooquery for {self.ticker}. Trying yfinance.")
                stock_yf = yf.Ticker(self.ticker)
                info = stock_yf.info
                self.fundamental_data = {
                    'debtToEquity': info.get('debtToEquity', 1.0),
                    'returnOnEquity': info.get('returnOnEquity', 0.15),
                    'dividendYield': info.get('dividendYield', 0.01),
                    'trailingPE': info.get('trailingPE', 20.0),
                    'priceToBook': info.get('priceToBook', 2.0),
                    'marketCap': info.get('marketCap', 1e9),
                    'netMargin': info.get('profitMargins', 0.1),
                    'returnOnCapitalEmployed': info.get('returnOnAssets', 0.05)  # Approximation
                }
            self.calculation_details.append(f"Fetched fundamental data for {self.ticker}")
            time.sleep(1)  # Delay to avoid API rate limits
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {self.ticker}: {e}")
            self.fundamental_data = {
                'debtToEquity': 1.0,
                'returnOnEquity': 0.15,
                'dividendYield': 0.01,
                'trailingPE': 20.0,
                'priceToBook': 2.0,
                'marketCap': 1e9,
                'netMargin': 0.1,
                'returnOnCapitalEmployed': 0.05
            }
            self.calculation_details.append(f"Error fetching fundamental data for {self.ticker}: {str(e)}. Using defaults.")

    def prompt_user_for_inputs(self):
        """
        Prompt user for parameters interactively (used for single-stock mode).
        """
        session = PromptSession(multiline=False)
        print(f"\nPlease provide parameters for {self.ticker} ({self.exchange}). Instructions are provided.\n")

        # Promoter Holding
        ownership_term = "Promoter Holding" if self.exchange == 'NSE' else "Insider Ownership"
        print(f"{ownership_term} (%): Percentage of shares owned by promoters (NSE) or insiders (other exchanges).")
        print(f"  - Source: Screener.in, Moneycontrol (NSE); SEC filings, Yahoo Finance (NASDAQ/NYSE).")
        print(f"  - Example: If promoters/insiders own 60%, enter 60.0. Default: 50.0")
        while True:
            value = session.prompt(f"Enter {ownership_term} (%) [0-100]: ", default="50.0")
            try:
                value = float(value)
                if 0 <= value <= 100:
                    self.user_inputs['promoter_holding'] = value
                    break
                print("Please enter a value between 0 and 100.")
            except ValueError:
                print("Please enter a valid number.")

        # Institutional Holding
        print(f"\nInstitutional Holding (%): Percentage owned by institutions (e.g., FII+DII for NSE, mutual funds for NASDAQ/NYSE).")
        print(f"  - Source: Screener.in, Moneycontrol (NSE); Yahoo Finance, SEC filings (NASDAQ/NYSE).")
        print(f"  - Example: Institutions own 35%, enter 35.0. Default: 15.0")
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
        print("\nProfit Growth YoY (%): Year-over-year net profit growth percentage.")
        print("  - Source: Screener.in, Moneycontrol, or annual reports.")
        print("  - Example: If profit grew 12%, enter 12.0. Default: 10.0")
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
        print("\nProfit CAGR 5Y (%): Compound Annual Growth Rate of net profit over 5 years.")
        print("  - Source: Screener.in, Moneycontrol, or calculate from annual reports.")
        print("  - Example: If CAGR is 8%, enter 8.0. Default: 12.0")
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

        # Economic Moat
        print("\nEconomic Moat: Competitive advantage (None, Narrow, Wide).")
        print("  - Source: Estimate based on brand, patents, or analyst reports.")
        print("  - Example: Wide for strong brands, Narrow for moderate advantages, None for no advantage. Default: None")
        while True:
            value = session.prompt("Enter Economic Moat (None, Narrow, Wide) [default: None]: ", default="None").strip().capitalize()
            if value in ['None', 'Narrow', 'Wide', '']:
                self.user_inputs['moat'] = value if value else None
                break
            print("Please enter None, Narrow, or Wide.")

    def set_user_inputs(self, inputs_dict):
        """
        Set user inputs from Excel or other sources.
        Args:
            inputs_dict (dict): Dictionary with input values.
        """
        defaults = {
            'promoter_holding': 50.0,
            'inst_holding': 15.0,
            'profit_growth': 10.0,
            'cagr': 12.0,
            'moat': None
        }
        for key, default in defaults.items():
            value = inputs_dict.get(key)
            if key == 'moat':
                if pd.isna(value) or value is None or value == '':
                    self.user_inputs[key] = default
                elif str(value).capitalize() in ['None', 'Narrow', 'Wide']:
                    self.user_inputs[key] = str(value).capitalize()
                else:
                    self.user_inputs[key] = default
                    self.calculation_details.append(f"Invalid moat value for {self.ticker}: {value}. Using default: {default}")
            else:
                try:
                    self.user_inputs[key] = float(value) if not pd.isna(value) else default
                except (ValueError, TypeError):
                    self.user_inputs[key] = default
                    self.calculation_details.append(f"Invalid {key} value for {self.ticker}: {value}. Using default: {default}")

    def calculate_technical_metrics(self):
        """Calculate technical metrics using yfinance data."""
        try:
            # Fetch historical data
            data = yf.Ticker(self.ticker).history(period=f"{max(self.params['dma_length'], self.params['rs_length'])}d")
            if data.empty:
                raise ValueError("No historical data available")

            # Price-based metrics
            self.technical_data['current_price'] = data['Close'][-1]
            self.technical_data['dma_value'] = data['Close'].rolling(window=self.params['dma_length']).mean()[-1]

            # Volume-based metrics
            self.technical_data['current_volume'] = data['Volume'][-1]
            self.technical_data['vol_avg'] = data['Volume'].rolling(window=self.params['vol_avg_length']).mean()[-1]

            # RSI
            rsi_indicator = RSIIndicator(close=data['Close'], window=self.params['rsi_length'])
            self.technical_data['rsi'] = rsi_indicator.rsi()[-1]

            # MACD
            macd_indicator = MACD(close=data['Close'])
            self.technical_data['macd'] = macd_indicator.macd_diff()[-1]

            # Relative Strength vs. Benchmark
            benchmark_data = yf.Ticker(self.benchmark_ticker).history(period=f"{self.params['rs_length']}d")
            if not benchmark_data.empty:
                stock_return = (data['Close'][-1] - data['Close'][0]) / data['Close'][0] * 100
                benchmark_return = (benchmark_data['Close'][-1] - benchmark_data['Close'][0]) / benchmark_data['Close'][0] * 100
                self.technical_data['rs'] = 50 + (stock_return - benchmark_return)
            else:
                self.technical_data['rs'] = 50.0
                self.calculation_details.append(f"No benchmark data for {self.benchmark_ticker}. Default RS score: 50")

            # Volatility
            daily_returns = data['Close'].pct_change().dropna()
            self.technical_data['volatility'] = daily_returns.std() * np.sqrt(252)  # Annualized
            self.calculation_details.append(f"Calculated technical metrics for {self.ticker}")
        except Exception as e:
            logger.error(f"Error calculating technical metrics for {self.ticker}: {e}")
            self.technical_data = {
                'current_price': 1000.0,
                'dma_value': 1000.0,
                'current_volume': 1e6,
                'vol_avg': 1e6,
                'rsi': 50.0,
                'macd': 0.0,
                'rs': 50.0,
                'volatility': 0.2
            }
            self.calculation_details.append(f"Error calculating technical metrics: {str(e)}. Using defaults.")

    def calculate_final_rating(self):
        """
        Calculate scores for all parameters and compute Final Rating.
        Returns:
            tuple: (scores list, final_rating)
        """
        scores = []

        # 1. Promoter Holding
        promoter_holding = self.user_inputs.get('promoter_holding', 50.0)
        promoter_score = min(100, promoter_holding * 2)
        scores.append(promoter_score)
        self.calculation_details.append(f"Promoter Holding: {promoter_holding}% -> Score: {promoter_score}")

        # 2. Institutional Holding
        inst_holding = self.user_inputs.get('inst_holding', 15.0)
        inst_score = min(100, inst_holding * 4)
        scores.append(inst_score)
        self.calculation_details.append(f"Institutional Holding: {inst_holding}% -> Score: {inst_score}")

        # 3. Debt-to-Equity Ratio
        de_ratio = self.fundamental_data.get('debtToEquity', 1.0)
        if isinstance(de_ratio, (int, float)):
            de_score = max(0, 100 - de_ratio * 50)
        else:
            de_score = 50.0
            self.calculation_details.append(f"Invalid D/E ratio: {de_ratio}. Using default score: 50")
        scores.append(de_score)
        self.calculation_details.append(f"Debt-to-Equity Ratio: {de_ratio} -> Score: {de_score}")

        # 4. Return on Equity
        roe = self.fundamental_data.get('returnOnEquity', 0.15)
        if isinstance(roe, (int, float)):
            roe_score = min(100, roe * 500)  # Convert to percentage and scale
        else:
            roe_score = 50.0
            self.calculation_details.append(f"Invalid RoE: {roe}. Using default score: 50")
        scores.append(roe_score)
        self.calculation_details.append(f"Return on Equity: {roe*100:.2f}% -> Score: {roe_score}")

        # 5. Profit Growth YoY
        profit_growth = self.user_inputs.get('profit_growth', 10.0)
        profit_growth_score = min(100, max(0, profit_growth * 5))
        scores.append(profit_growth_score)
        self.calculation_details.append(f"Profit Growth YoY: {profit_growth}% -> Score: {profit_growth_score}")

        # 6. Profit CAGR 5Y
        cagr = self.user_inputs.get('cagr', 12.0)
        cagr_score = min(100, max(0, cagr * 5))
        scores.append(cagr_score)
        self.calculation_details.append(f"Profit CAGR 5Y: {cagr}% -> Score: {cagr_score}")

        # 7. Dividend Yield
        div_yield = self.fundamental_data.get('dividendYield', 0.01)
        if isinstance(div_yield, (int, float)):
            div_yield_score = min(100, div_yield * 5000)  # Convert to percentage and scale
        else:
            div_yield_score = 50.0
            self.calculation_details.append(f"Invalid Dividend Yield: {div_yield}. Using default score: 50")
        scores.append(div_yield_score)
        self.calculation_details.append(f"Dividend Yield: {div_yield*100:.2f}% -> Score: {div_yield_score}")

        # 8. Economic Moat
        moat = self.user_inputs.get('moat')
        if moat is None:
            # Calculate moat if not provided
            net_margin = self.fundamental_data.get('netMargin', 0.1)
            market_cap = self.fundamental_data.get('marketCap', 1e9)
            roce = self.fundamental_data.get('returnOnCapitalEmployed', 0.05)
            revenue_stability = 0.1  # Placeholder
            moat_score = 0
            self.moat_basis = "No moat due to: "
            reasons = []
            if net_margin > 0.15:
                moat_score += 25
                reasons.append("High profit margin")
            if market_cap > 5e9:
                moat_score += 25
                reasons.append("Large market cap")
            if roce > 0.1:
                moat_score += 25
                reasons.append("High ROCE")
            if revenue_stability > 0.2:
                moat_score += 25
                reasons.append("Stable revenue")
            if moat_score >= 75:
                moat = "Wide"
                moat_score = 100
            elif moat_score >= 50:
                moat = "Narrow"
                moat_score = 50
            else:
                moat = "None"
                moat_score = 0
            if reasons:
                self.moat_basis = f"{moat} moat due to: {', '.join(reasons)}"
            else:
                self.moat_basis = "No moat: Insufficient competitive advantage"
        else:
            moat_score = {'Wide': 100, 'Narrow': 50, 'None': 0}.get(moat, 0)
            self.moat_basis = f"{moat} (Overridden by user to {moat})"
        scores.append(moat_score)
        self.calculation_details.append(f"Economic Moat: {self.moat_basis} -> Score: {moat_score}")

        # 9. Valuation Score
        pe_ratio = self.fundamental_data.get('trailingPE', 20.0)
        pb_ratio = self.fundamental_data.get('priceToBook', 2.0)
        industry_pe = 20.0  # Placeholder
        pe_score = max(0, 100 - (pe_ratio / industry_pe) * 50)
        pb_score = max(0, 100 - (pb_ratio / 5) * 50)
        valuation_score = (pe_score + pb_score) / 2
        scores.append(valuation_score)
        self.calculation_details.append(f"Valuation Score: PE={pe_ratio:.2f}, PB={pb_ratio:.2f} -> Score: {valuation_score}")

        # 10. SymbolTrendRS
        rsi = self.technical_data.get('rsi', 50.0)
        macd = self.technical_data.get('macd', 0.0)
        rsi_score = (rsi / 100) * 50
        macd_score = 50 if macd > 0 else 25
        symbol_trend_score = (rsi_score + macd_score) / 2
        scores.append(symbol_trend_score)
        self.calculation_details.append(f"SymbolTrendRS: RSI={rsi:.2f}, MACD={'Positive' if macd > 0 else 'Negative'} -> Score: {symbol_trend_score}")

        # 11. NIFTY200DMARSIVolume
        nifty_score = 0
        # DMA component
        if self.technical_data.get('current_price', 1000.0) > self.technical_data.get('dma_value', 1000.0):
            nifty_score += 25
            self.calculation_details.append("NIFTY200DMARSIVolume: Price > 200-Day DMA (+25)")
        else:
            self.calculation_details.append("NIFTY200DMARSIVolume: Price <= 200-Day DMA (+0)")
        # RSI component
        if 30 <= rsi <= 70:
            nifty_score += 25
            self.calculation_details.append("NIFTY200DMARSIVolume: RSI in neutral range (+25)")
        else:
            nifty_score += 10
            self.calculation_details.append("NIFTY200DMARSIVolume: RSI outside neutral range (+10)")
        # Volume component
        if self.technical_data.get('current_volume', 1e6) > self.technical_data.get('vol_avg', 1e6):
            nifty_score += 25
            self.calculation_details.append("NIFTY200DMARSIVolume: Volume > 20-Day Vol MA (+25)")
        else:
            nifty_score += 10
            self.calculation_details.append("NIFTY200DMARSIVolume: Volume <= 20-Day Vol MA (+10)")
        # RS component
        rs = self.technical_data.get('rs', 50.0)
        if rs > 50:
            nifty_score += 25
            self.calculation_details.append("NIFTY200DMARSIVolume: RS > 50 (+25)")
        else:
            nifty_score += 10
            self.calculation_details.append("NIFTY200DMARSIVolume: RS <= 50 (+10)")
        scores.append(nifty_score)
        self.calculation_details.append(f"NIFTY200DMARSIVolume: Total Score: {nifty_score}")

        # 12. 200-Day DMA
        dma_score = 100 if self.technical_data.get('current_price', 1000.0) > self.technical_data.get('dma_value', 1000.0) else 0
        scores.append(dma_score)
        self.calculation_details.append(f"200-Day DMA: {'Above' if dma_score == 100 else 'Below'} -> Score: {dma_score}")

        # 13. 14-Day RSI
        rsi_score = 50
        if rsi < 30:
            rsi_score = 25
            self.calculation_details.append("14-Day RSI: Oversold (<30) -> Score: 25")
        elif rsi > 70:
            rsi_score = 75
            self.calculation_details.append("14-Day RSI: Overbought (>70) -> Score: 75")
        else:
            self.calculation_details.append("14-Day RSI: Neutral -> Score: 50")
        scores.append(rsi_score)

        # 14. 20-Day Volume MA
        vol_score = 75 if self.technical_data.get('current_volume', 1e6) > self.technical_data.get('vol_avg', 1e6) else 25
        scores.append(vol_score)
        self.calculation_details.append(f"20-Day Volume MA: {'Above' if vol_score == 75 else 'Below'} -> Score: {vol_score}")

        # 15. Simple RS vs. Benchmark
        rs_score = max(0, min(100, self.technical_data.get('rs', 50.0)))
        scores.append(rs_score)
        self.calculation_details.append(f"Simple RS vs {self.benchmark_ticker}: {rs_score:.2f} -> Score: {rs_score}")

        # Apply volatility penalty
        volatility = self.technical_data.get('volatility', 0.2)
        volatility_penalty = max(0, 100 - (volatility * 100)) / 100
        self.calculation_details.append(f"Volatility: {volatility:.2f} -> Penalty: {volatility_penalty:.2f}")

        # Calculate Final Rating
        weighted_scores = [score * weight * volatility_penalty for score, weight in zip(scores, self.weights)]
        self.final_rating = round(sum(weighted_scores) / sum(self.weights), 2)
        self.calculation_details.append(f"Final Rating: {self.final_rating}")

        return scores, self.final_rating

    def get_results(self):
        """
        Return results as a dictionary for output.
        Returns:
            dict: Results including ticker, scores, and final rating.
        """
        return {
            'Ticker': self.ticker,
            'Final Rating (0-100)': self.final_rating,
            'Promoter Holding': f"{self.user_inputs.get('promoter_holding', 50.0)}% -> {min(100, self.user_inputs.get('promoter_holding', 50.0) * 2)}",
            'Inst. Holding': f"{self.user_inputs.get('inst_holding', 15.0)}% -> {min(100, self.user_inputs.get('inst_holding', 15.0) * 4)}",
            'D/E Ratio': f"{self.fundamental_data.get('debtToEquity', 1.0)} -> {max(0, 100 - self.fundamental_data.get('debtToEquity', 1.0) * 50)}",
            'RoE': f"{self.fundamental_data.get('returnOnEquity', 0.15)*100:.2f}% -> {min(100, self.fundamental_data.get('returnOnEquity', 0.15) * 500)}",
            'Profit Growth YoY': f"{self.user_inputs.get('profit_growth', 10.0)}% -> {min(100, max(0, self.user_inputs.get('profit_growth', 10.0) * 5))}",
            'Profit CAGR 5Y': f"{self.user_inputs.get('cagr', 12.0)}% -> {min(100, max(0, self.user_inputs.get('cagr', 12.0) * 5))}",
            'Dividend Yield': f"{self.fundamental_data.get('dividendYield', 0.01)*100:.2f}% -> {min(100, self.fundamental_data.get('dividendYield', 0.01) * 5000)}",
            'Economic Moat': self.moat_basis,
            'Valuation Score': self.calculate_final_rating()[0][8],  # Get from scores
            'SymbolTrendRS': self.calculate_final_rating()[0][9],
            'NIFTY200DMARSIVolume': self.calculate_final_rating()[0][10],
            '200-Day DMA': self.calculate_final_rating()[0][11],
            '14-Day RSI': self.calculate_final_rating()[0][12],
            '20-Day Volume MA': self.calculate_final_rating()[0][13],
            'Simple RS vs Benchmark': self.calculate_final_rating()[0][14],
            'Calculation Details': '; '.join(self.calculation_details)
        }

    def display_table(self, scores, final_rating):
        """
        Display results in a tabulated format.
        Args:
            scores (list): List of parameter scores.
            final_rating (float): Final Rating.
        """
        table = [
            ["Promoter Holding", f"{self.user_inputs.get('promoter_holding', 50.0)}%", scores[0]],
            ["Inst. Holding", f"{self.user_inputs.get('inst_holding', 15.0)}%", scores[1]],
            ["D/E Ratio", self.fundamental_data.get('debtToEquity', 1.0), scores[2]],
            ["RoE", f"{self.fundamental_data.get('returnOnEquity', 0.15)*100:.2f}%", scores[3]],
            ["Profit Growth YoY", f"{self.user_inputs.get('profit_growth', 10.0)}%", scores[4]],
            ["Profit CAGR 5Y", f"{self.user_inputs.get('cagr', 12.0)}%", scores[5]],
            ["Dividend Yield", f"{self.fundamental_data.get('dividendYield', 0.01)*100:.2f}%", scores[6]],
            ["Economic Moat", self.moat_basis, scores[7]],
            ["Valuation Score", f"PE={self.fundamental_data.get('trailingPE', 20.0):.2f}, PB={self.fundamental_data.get('priceToBook', 2.0):.2f}", scores[8]],
            ["SymbolTrendRS", f"RSI={self.technical_data.get('rsi', 50.0):.2f}", scores[9]],
            ["NIFTY200DMARSIVolume", "Composite", scores[10]],
            ["200-Day DMA", "Above" if scores[11] == 100 else "Below", scores[11]],
            ["14-Day RSI", f"{self.technical_data.get('rsi', 50.0):.2f}", scores[12]],
            ["20-Day Volume MA", "Above" if scores[13] == 75 else "Below", scores[13]],
            ["Simple RS vs Benchmark", f"{self.technical_data.get('rs', 50.0):.2f}", scores[14]],
            ["Final Rating", "", final_rating]
        ]
        print(f"\nAnalysis for {self.ticker} ({self.exchange}):")
        print(tabulate(table, headers=["Parameter", "Value", "Score"], tablefmt="grid"))

        # Save to CSV
        df = pd.DataFrame(table, columns=["Parameter", "Value", "Score"])
        df.to_csv(f"{self.ticker}_analysis.csv", index=False)
        logger.info(f"Saved analysis to {self.ticker}_analysis.csv")

    def plot_metrics(self, plot_dir="plots"):
        """
        Plot stock price and 200-Day DMA.
        Args:
            plot_dir (str): Directory to save plots.
        """
        try:
            data = yf.Ticker(self.ticker).history(period=f"{self.params['dma_length']+50}d")
            if data.empty:
                raise ValueError("No data for plotting")

            plt.figure(figsize=(10, 6))
            plt.plot(data.index, data['Close'], label='Close Price')
            plt.plot(data.index, data['Close'].rolling(window=self.params['dma_length']).mean(), label='200-Day DMA')
            plt.title(f"{self.ticker} Price and 200-Day DMA")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.grid()

            safe_ticker = self.ticker.replace('/', '_').replace('^', '_')
            plt.savefig(f"{plot_dir}/{safe_ticker}_plot.png")
            plt.close()
            logger.info(f"Saved plot to {plot_dir}/{safe_ticker}_plot.png")
        except Exception as e:
            logger.error(f"Error plotting for {self.ticker}: {e}")
            self.calculation_details.append(f"Error plotting: {str(e)}")

def process_portfolio(input_excel, output_excel="portfolio_rankings.xlsx"):
    """
    Process a portfolio from an Excel file and generate an output Excel with rankings.
    Args:
        input_excel (str): Path to input Excel file.
        output_excel (str): Path to output Excel file.
    """
    try:
        # Read input Excel
        df = pd.read_excel(input_excel)
        required_columns = ['Ticker']
        optional_columns = ['Promoter Holding (%)', 'Inst. Holding (%)', 'Profit Growth YoY (%)',
                           'Profit CAGR 5Y (%)', 'Economic Moat', 'Exchange']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Input Excel must contain 'Ticker' column")

        # Check for duplicate tickers
        if df['Ticker'].duplicated().any():
            duplicates = df[df['Ticker'].duplicated(keep=False)]['Ticker'].unique()
            logger.warning(f"Duplicate tickers found: {', '.join(duplicates)}. Processing first occurrence only.")
            df = df.drop_duplicates(subset=['Ticker'], keep='first')

        # Create plots directory upfront
        plot_dir = "plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        results = []
        all_calculation_details = []
        missing_inputs = []

        # Process each stock
        for index, row in df.iterrows():
            ticker = str(row['Ticker']).strip().upper()
            # Get exchange from Excel, default to NSE
            exchange = row.get('Exchange', 'NSE').strip().upper() if 'Exchange' in df.columns else 'NSE'
            
            # Validate exchange
            if exchange not in SUPPORTED_EXCHANGES:
                logger.error(f"Unsupported exchange for {ticker}: {exchange}. Supported exchanges: {', '.join(SUPPORTED_EXCHANGES)}")
                results.append({
                    'Ticker': ticker,
                    'Final Rating (0-100)': 'Error',
                    'Calculation Details': f"Error: Unsupported exchange {exchange}"
                })
                continue

            # Append suffix if defined for the exchange
            suffix = config[exchange].get('suffix', '')
            if suffix and not ticker.endswith(suffix):
                ticker += suffix

            print(f"\nProcessing {ticker} ({exchange})...")
            try:
                analyzer = StockAnalyzer(ticker, exchange=exchange)
                analyzer.fetch_fundamental_data()

                # Track missing inputs
                inputs_dict = {
                    'promoter_holding': row.get('Promoter Holding (%)', np.nan),
                    'inst_holding': row.get('Inst. Holding (%)', np.nan),
                    'profit_growth': row.get('Profit Growth YoY (%)', np.nan),
                    'cagr': row.get('Profit CAGR 5Y (%)', np.nan),
                    'moat': row.get('Economic Moat', None)
                }
                missing = [key for key, value in inputs_dict.items() if pd.isna(value) and key != 'moat']
                if missing:
                    missing_inputs.append(f"{ticker}: Missing {', '.join(missing)}")

                # Set user inputs from Excel
                analyzer.set_user_inputs(inputs_dict)

                # Calculate metrics and rating
                analyzer.calculate_technical_metrics()
                scores, final_rating = analyzer.calculate_final_rating()
                result = analyzer.get_results()
                results.append(result)
                all_calculation_details.extend([f"{ticker}: {detail}" for detail in analyzer.calculation_details])

                # Generate plot
                analyzer.plot_metrics(plot_dir)

            except Exception as e:
                logger.error(f"Failed to process {ticker}: {e}")
                results.append({
                    'Ticker': ticker,
                    'Final Rating (0-100)': 'Error',
                    'Calculation Details': f"Error: {str(e)}. Check ticker validity or try again later."
                })
                all_calculation_details.append(f"{ticker}: Error: {str(e)}")

        # Log missing inputs summary
        if missing_inputs:
            print("\nWarning: Missing inputs in Excel file (defaults used):")
            for entry in missing_inputs:
                print(f"- {entry}")

        # Create output Excel
        results_df = pd.DataFrame(results)
        calc_details_df = pd.DataFrame(all_calculation_details, columns=['Calculation Details'])

        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Rankings', index=False)
            calc_details_df.to_excel(writer, sheet_name='Calculation Details', index=False)

            # Auto-adjust column widths
            for sheet in ['Rankings', 'Calculation Details']:
                worksheet = writer.sheets[sheet]
                for col in worksheet.columns:
                    max_length = 0
                    column = col[0].column_letter
                    for cell in col:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = max_length + 2
                    worksheet.column_dimensions[column].width = adjusted_width

        logger.info(f"Saved portfolio rankings to {output_excel}")
        print(f"\nResults saved to {output_excel}")

    except Exception as e:
        logger.error(f"Failed to process portfolio: {e}")
        print(f"Error: {e}")

def main():
    """Main function to run stock or portfolio analysis."""
    parser = argparse.ArgumentParser(description="Stock Ranking Analysis Tool")
    parser.add_argument('--portfolio', type=str, help="Path to portfolio Excel file (e.g., portfolio.xlsx)")
    args = parser.parse_args()

    print("Disclaimer: This tool is for educational purposes only and not financial advice. \nThe developer is not liable for any losses or actions by SEBI or other regulators. \nUsers are responsible for their investment decisions and should consult a registered advisor.\n")

    if args.portfolio:
        # Portfolio mode
        if not os.path.exists(args.portfolio):
            print(f"Error: Portfolio file {args.portfolio} not found.")
            return
        process_portfolio(args.portfolio)
    else:
        # Single stock mode
        session = PromptSession(multiline=False)
        print(f"Supported exchanges: {', '.join(SUPPORTED_EXCHANGES)}")
        ticker = session.prompt("Enter the stock ticker (e.g., RELIANCE.NS for NSE, NXPI for NASDAQ, AAPL for NYSE): ").strip().upper()
        exchange = session.prompt(f"Enter the exchange (default NSE): ", default="NSE").strip().upper()
        
        if exchange not in SUPPORTED_EXCHANGES:
            print(f"Error: Unsupported exchange {exchange}. Supported exchanges: {', '.join(SUPPORTED_EXCHANGES)}")
            return
        
        # Append suffix if needed
        suffix = config[exchange].get('suffix', '')
        if suffix and not ticker.endswith(suffix):
            ticker += suffix

        try:
            analyzer = StockAnalyzer(ticker, exchange=exchange)
            analyzer.fetch_fundamental_data()
            analyzer.prompt_user_for_inputs()
            analyzer.calculate_technical_metrics()
            scores, final_rating = analyzer.calculate_final_rating()
            analyzer.display_table(scores, final_rating)
            analyzer.plot_metrics()
        except Exception as e:
            logger.error(f"Analysis failed for {ticker}: {e}")
            print("Please ensure the ticker and exchange are valid and try again.")

if __name__ == "__main__":
    main()

