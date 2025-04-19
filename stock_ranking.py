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
import requests
from bs4 import BeautifulSoup
import json

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
        self.recommendation = None
        self.recommendation_reasons = []

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
            # Implement rate-limiting mechanism to avoid API rate limits
            if hasattr(self, '_last_api_call') and time.time() - self._last_api_call < 1:
                time.sleep(1 - (time.time() - self._last_api_call))
            self._last_api_call = time.time()
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {self.ticker}: {e}")
            self.fundamental_data = {
                'debtToEquity': float(config['DEFAULT'].get('default_debtToEquity', 1.0)),
                'returnOnEquity': float(config['DEFAULT'].get('default_returnOnEquity', 0.15)),
                'dividendYield': float(config['DEFAULT'].get('default_dividendYield', 0.01)),
                'trailingPE': float(config['DEFAULT'].get('default_trailingPE', 20.0)),
                'priceToBook': float(config['DEFAULT'].get('default_priceToBook', 2.0)),
                'marketCap': float(config['DEFAULT'].get('default_marketCap', 1e9)),
                'netMargin': float(config['DEFAULT'].get('default_netMargin', 0.1)),
                'returnOnCapitalEmployed': float(config['DEFAULT'].get('default_returnOnCapitalEmployed', 0.05))
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
        print("  - Example: Wide for strong brands, Narrow for moderate advantages, None for no advantage.")

        # Real-Time Moat Calculation
        net_margin = self.fundamental_data.get('netMargin', 0.1)
        market_cap = self.fundamental_data.get('marketCap', 1e9)
        roce = self.fundamental_data.get('returnOnCapitalEmployed', 0.05)
        revenue_stability = 0.1  # Placeholder
        brand_strength = 0
        patent_count = 0

        try:
            yf_ticker = yf.Ticker(self.ticker)
            info = yf_ticker.info
            if 'revenueGrowth' in info and info['revenueGrowth'] is not None:
                revenue_stability = abs(info['revenueGrowth'])
            if 'longBusinessSummary' in info and info['longBusinessSummary']:
                summary = info['longBusinessSummary'].lower()
                if any(word in summary for word in ['brand', 'trusted', 'leader', 'premium']):
                    brand_strength = 1
            if 'sector' in info and info['sector']:
                if info['sector'].lower() in ['technology', 'pharmaceuticals', 'semiconductors']:
                    patent_count = 1
        except Exception as e:
            logger.warning(f"Could not fetch extra real-time info for moat: {e}")

        moat_score = 0
        reasons = []
        if net_margin > 0.18:
            moat_score += 20
            reasons.append("Very high profit margin")
        elif net_margin > 0.12:
            moat_score += 10
            reasons.append("Good profit margin")
        if market_cap > 10e9:
            moat_score += 20
            reasons.append("Very large market cap")
        elif market_cap > 2e9:
            moat_score += 10
            reasons.append("Large market cap")
        if roce > 0.15:
            moat_score += 20
            reasons.append("Excellent ROCE")
        elif roce > 0.10:
            moat_score += 10
            reasons.append("Good ROCE")
        if revenue_stability > 0.15:
            moat_score += 15
            reasons.append("Strong revenue stability")
        elif revenue_stability > 0.08:
            moat_score += 7
            reasons.append("Moderate revenue stability")
        if brand_strength > 0:
            moat_score += 10
            reasons.append("Recognized brand strength")
        if patent_count > 0:
            moat_score += 5
            reasons.append("Patent portfolio")
        if 'sector' in locals() and info.get('sector', '').lower() in ['utilities', 'energy']:
            moat_score += 5
            reasons.append("Regulated/utility sector")

        if moat_score >= 60:
            calculated_moat = "Wide"
            moat_reason = f"Calculated Wide moat due to: {', '.join(reasons)}"
        elif moat_score >= 30:
            calculated_moat = "Narrow"
            moat_reason = f"Calculated Narrow moat due to: {', '.join(reasons)}"
        else:
            calculated_moat = "None"
            moat_reason = "Calculated No moat: Insufficient competitive advantage"

        print(f"\nBased on real-time and fundamental data, a preliminary economic moat assessment suggests: {calculated_moat}.")
        print(f"Reason: {moat_reason}")
        print("If you have more qualitative insights (e.g., strong brand, patents, network effects), consider overriding this assessment.")

        while True:
            override = session.prompt("Do you agree with this assessment? (yes/no/override) [default: yes]: ", default="yes").strip().lower()
            if override in ['yes', '']:
                self.user_inputs['moat'] = calculated_moat
                self.moat_basis = moat_reason
                break
            elif override == 'no':
                self.user_inputs['moat'] = calculated_moat
                self.moat_basis = moat_reason
                break
            elif override == 'override':
                while True:
                    value = session.prompt("Enter Economic Moat (None, Narrow, Wide) [default: None]: ", default="None").strip().capitalize()
                    if value in ['None', 'Narrow', 'Wide', '']:
                        self.user_inputs['moat'] = value if value else None
                        self.moat_basis = f"Overridden by user to {value}"
                        break
                    print("Please enter None, Narrow, or Wide.")
                break
            else:
                print("Please enter yes, no, or override.")

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
            data = yf.Ticker(self.ticker).history(period=f"{max(self.params['dma_length'], self.params['rs_length'])}d")
            if data.empty:
                raise ValueError("No historical data available")

            self.technical_data['current_price'] = data['Close'].iloc[-1]
            self.technical_data['dma_value'] = data['Close'].rolling(window=self.params['dma_length']).mean().iloc[-1]
            self.technical_data['current_volume'] = data['Volume'].iloc[-1]
            self.technical_data['vol_avg'] = data['Volume'].rolling(window=self.params['vol_avg_length']).mean().iloc[-1]

            rsi_indicator = RSIIndicator(close=data['Close'], window=self.params['rsi_length'])
            self.technical_data['rsi'] = rsi_indicator.rsi().iloc[-1]

            macd_indicator = MACD(close=data['Close'])
            self.technical_data['macd'] = macd_indicator.macd_diff().iloc[-1]

            benchmark_data = yf.Ticker(self.benchmark_ticker).history(period=f"{self.params['rs_length']}d")
            if not benchmark_data.empty:
                stock_return = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100
                benchmark_return = (benchmark_data['Close'].iloc[-1] - benchmark_data['Close'].iloc[0]) / benchmark_data['Close'].iloc[0] * 100
                self.technical_data['rs'] = 50 + (stock_return - benchmark_return)
            else:
                self.technical_data['rs'] = 50.0
                self.calculation_details.append(f"No benchmark data for {self.benchmark_ticker}. Default RS score: 50")

            daily_returns = data['Close'].pct_change().dropna()
            self.technical_data['volatility'] = daily_returns.std() * np.sqrt(252)
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

        """Calculate technical metrics using yfinance data."""
        try:
            data = yf.Ticker(self.ticker).history(period=f"{max(self.params['dma_length'], self.params['rs_length'])}d")
            if data.empty:
                raise ValueError("No historical data available")

            self.technical_data['current_price'] = data['Close'][-1]
            self.technical_data['dma_value'] = data['Close'].rolling(window=self.params['dma_length']).mean()[-1]
            self.technical_data['current_volume'] = data['Volume'][-1]
            self.technical_data['vol_avg'] = data['Volume'].rolling(window=self.params['vol_avg_length']).mean()[-1]

            rsi_indicator = RSIIndicator(close=data['Close'], window=self.params['rsi_length'])
            self.technical_data['rsi'] = rsi_indicator.rsi()[-1]

            macd_indicator = MACD(close=data['Close'])
            self.technical_data['macd'] = macd_indicator.macd_diff()[-1]

            benchmark_data = yf.Ticker(self.benchmark_ticker).history(period=f"{self.params['rs_length']}d")
            if not benchmark_data.empty:
                stock_return = (data['Close'][-1] - data['Close'][0]) / data['Close'][0] * 100
                benchmark_return = (benchmark_data['Close'][-1] - benchmark_data['Close'][0]) / benchmark_data['Close'][0] * 100
                self.technical_data['rs'] = 50 + (stock_return - benchmark_return)
            else:
                self.technical_data['rs'] = 50.0
                self.calculation_details.append(f"No benchmark data for {self.benchmark_ticker}. Default RS score: 50")

            daily_returns = data['Close'].pct_change().dropna()
            self.technical_data['volatility'] = daily_returns.std() * np.sqrt(252)
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
        calculation_details = []

        # 1. Promoter Holding
        promoter_holding = self.user_inputs.get('promoter_holding', 50.0)
        promoter_score = min(100, promoter_holding * 2)
        scores.append(promoter_score)
        calculation_details.append(f"Promoter Holding: {promoter_holding}% -> Score: {promoter_score}")

        # 2. Institutional Holding
        inst_holding = self.user_inputs.get('inst_holding', 15.0)
        inst_score = min(100, inst_holding * 4)
        scores.append(inst_score)
        calculation_details.append(f"Institutional Holding: {inst_holding}% -> Score: {inst_score}")

        # 3. Debt-to-Equity Ratio
        de_ratio = self.fundamental_data.get('debtToEquity', 1.0)
        if isinstance(de_ratio, (int, float)):
            de_score = max(0, 100 - de_ratio * 50)
        else:
            de_score = 50.0
            calculation_details.append(f"Invalid D/E ratio: {de_ratio}. Using default score: 50")
        scores.append(de_score)
        calculation_details.append(f"Debt-to-Equity Ratio: {de_ratio} -> Score: {de_score}")

        # 4. Return on Equity
        roe = self.fundamental_data.get('returnOnEquity', 0.15)
        if isinstance(roe, (int, float)):
            roe_score = min(100, roe * 500)
        else:
            roe_score = 50.0
            calculation_details.append(f"Invalid RoE: {roe}. Using default score: 50")
        scores.append(roe_score)
        calculation_details.append(f"Return on Equity: {roe*100:.2f}% -> Score: {roe_score}")

        # 5. Profit Growth YoY
        profit_growth = self.user_inputs.get('profit_growth', 10.0)
        profit_growth_score = min(100, max(0, profit_growth * 5))
        scores.append(profit_growth_score)
        calculation_details.append(f"Profit Growth YoY: {profit_growth}% -> Score: {profit_growth_score}")

        # 6. Profit CAGR 5Y
        cagr = self.user_inputs.get('cagr', 12.0)
        cagr_score = min(100, max(0, cagr * 5))
        scores.append(cagr_score)
        calculation_details.append(f"Profit CAGR 5Y: {cagr}% -> Score: {cagr_score}")

        # 7. Dividend Yield
        div_yield = self.fundamental_data.get('dividendYield', 0.01)
        if isinstance(div_yield, (int, float)):
            div_yield_score = min(100, div_yield * 5000)
        else:
            div_yield_score = 50.0
            calculation_details.append(f"Invalid Dividend Yield: {div_yield}. Using default score: 50")
        scores.append(div_yield_score)
        calculation_details.append(f"Dividend Yield: {div_yield*100:.2f}% -> Score: {div_yield_score}")

        # 8. Economic Moat
        moat = self.user_inputs.get('moat')
        if moat is None:
            net_margin = self.fundamental_data.get('netMargin', 0.1)
            market_cap = self.fundamental_data.get('marketCap', 1e9)
            roce = self.fundamental_data.get('returnOnCapitalEmployed', 0.05)
            revenue_stability = 0.1
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
        calculation_details.append(f"Economic Moat: {self.moat_basis} -> Score: {moat_score}")

        # 9. Valuation Score
        pe_ratio = self.fundamental_data.get('trailingPE', 20.0)
        pb_ratio = self.fundamental_data.get('priceToBook', 2.0)
        industry_pe = 20.0
        pe_score = max(0, 100 - (pe_ratio / industry_pe) * 50)
        pb_score = max(0, 100 - (pb_ratio / 5) * 50)
        valuation_score = (pe_score + pb_score) / 2
        scores.append(valuation_score)
        calculation_details.append(f"Valuation Score: PE={pe_ratio:.2f}, PB={pb_ratio:.2f} -> Score: {valuation_score}")

        # 10. SymbolTrendRS
        rsi = self.technical_data.get('rsi', 50.0)
        macd = self.technical_data.get('macd', 0.0)
        rsi_score = (rsi / 100) * 50
        macd_score = 50 if macd > 0 else 25
        symbol_trend_score = (rsi_score + macd_score) / 2
        scores.append(symbol_trend_score)
        calculation_details.append(f"SymbolTrendRS: RSI={rsi:.2f}, MACD={'Positive' if macd > 0 else 'Negative'} -> Score: {symbol_trend_score}")

        # 11. NIFTY200DMARSIVolume
        nifty_score = 0
        if self.technical_data.get('current_price', 1000.0) > self.technical_data.get('dma_value', 1000.0):
            nifty_score += 25
            calculation_details.append("NIFTY200DMARSIVolume: Price > 200-Day DMA (+25)")
        else:
            calculation_details.append("NIFTY200DMARSIVolume: Price <= 200-Day DMA (+0)")
        if 30 <= rsi <= 70:
            nifty_score += 25
            calculation_details.append("NIFTY200DMARSIVolume: RSI in neutral range (+25)")
        else:
            nifty_score += 10
            calculation_details.append("NIFTY200DMARSIVolume: RSI outside neutral range (+10)")
        if self.technical_data.get('current_volume', 1e6) > self.technical_data.get('vol_avg', 1e6):
            nifty_score += 25
            calculation_details.append("NIFTY200DMARSIVolume: Volume > 20-Day Vol MA (+25)")
        else:
            nifty_score += 10
            calculation_details.append("NIFTY200DMARSIVolume: Volume <= 20-Day Vol MA (+10)")
        rs = self.technical_data.get('rs', 50.0)
        if rs > 50:
            nifty_score += 25
            calculation_details.append("NIFTY200DMARSIVolume: RS > 50 (+25)")
        else:
            nifty_score += 10
            calculation_details.append("NIFTY200DMARSIVolume: RS <= 50 (+10)")
        scores.append(nifty_score)
        calculation_details.append(f"NIFTY200DMARSIVolume: Total Score: {nifty_score}")

        # 12. 200-Day DMA
        dma_score = 100 if self.technical_data.get('current_price', 1000.0) > self.technical_data.get('dma_value', 1000.0) else 0
        scores.append(dma_score)
        calculation_details.append(f"200-Day DMA: {'Above' if dma_score == 100 else 'Below'} -> Score: {dma_score}")

        # 13. 14-Day RSI
        rsi_score = 50
        if rsi < 30:
            rsi_score = 25
            calculation_details.append("14-Day RSI: Oversold (<30) -> Score: 25")
        elif rsi > 70:
            rsi_score = 75
            calculation_details.append("14-Day RSI: Overbought (>70) -> Score: 75")
        else:
            calculation_details.append("14-Day RSI: Neutral -> Score: 50")
        scores.append(rsi_score)

        # 14. 20-Day Volume MA
        vol_score = 75 if self.technical_data.get('current_volume', 1e6) > self.technical_data.get('vol_avg', 1e6) else 25
        scores.append(vol_score)
        calculation_details.append(f"20-Day Volume MA: {'Above' if vol_score == 75 else 'Below'} -> Score: {vol_score}")

        # 15. Simple RS vs. Benchmark
        rs_score = max(0, min(100, self.technical_data.get('rs', 50.0)))
        scores.append(rs_score)
        calculation_details.append(f"Simple RS vs {self.benchmark_ticker}: {rs_score:.2f} -> Score: {rs_score}")

        # Apply tiered volatility penalty
        volatility = self.technical_data.get('volatility', 0.2)
        if volatility <= 0.2:
            volatility_penalty = 0.9
        elif volatility <= 0.4:
            volatility_penalty = 0.7
        elif volatility <= 0.6:
            volatility_penalty = 0.6
        else:
            volatility_penalty = 0.5
        calculation_details.append(f"Volatility: {volatility:.2f} -> Penalty: {volatility_penalty:.2f}")

        # Calculate Final Rating
        weighted_scores = [score * weight * volatility_penalty for score, weight in zip(scores, self.weights)]
        self.final_rating = round(sum(weighted_scores), 2)
        self.calculation_details.extend(calculation_details)
        return scores, self.final_rating

    def calculate_recommendation(self):
        """
        Calculate buy/add/sell recommendation based on final rating and key metrics.
        Returns:
            str: Recommendation (Buy, Add, Sell)
        """
        recommendation = "Hold"
        reasons = []

        if self.final_rating >= 80:
            recommendation = "Buy"
            reasons.append("High rating (>= 80)")
        elif 60 <= self.final_rating < 80:
            recommendation = "Add"
            reasons.append("Moderate rating (60-79)")
        elif self.final_rating < 60:
            recommendation = "Sell"
            reasons.append("Low rating (< 60)")

        rs = self.technical_data.get('rs', 50.0)
        if rs > 70:
            reasons.append("Strong RS > 70")
        elif rs < 30:
            recommendation = "Sell"
            reasons.append("Weak RS < 30")

        valuation_score = self.calculate_final_rating()[0][8]
        if valuation_score > 80:
            reasons.append("Attractive valuation (Valuation Score > 80)")
        elif valuation_score < 40:
            recommendation = "Sell"
            reasons.append("Overvalued (Valuation Score < 40)")

        rsi = self.technical_data.get('rsi', 50.0)
        current_price = self.technical_data.get('current_price', 1000.0)
        dma_value = self.technical_data.get('dma_value', 1000.0)
        current_volume = self.technical_data.get('current_volume', 1e6)
        vol_avg = self.technical_data.get('vol_avg', 1e6)

        try:
            data = yf.Ticker(self.ticker).history(period="60d")
            dma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
            dma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
        except Exception:
            dma_20 = dma_50 = current_price

        golden_crossover = current_price > dma_value
        high_volume_breakout = current_volume > vol_avg
        short_term_dma_crossover = dma_20 > dma_50

        if rsi < 30 and golden_crossover and high_volume_breakout and short_term_dma_crossover:
            recommendation = "Buy"
            reasons.append("Oversold (RSI < 30), Golden Crossover, High Volume Breakout, 20-DMA > 50-DMA")
        elif rsi < 30 and golden_crossover and high_volume_breakout:
            recommendation = "Buy"
            reasons.append("Oversold (RSI < 30), Golden Crossover, High Volume Breakout")
        elif rsi < 30:
            recommendation = "Buy"
            reasons.append("Oversold (RSI < 30)")
        elif rsi > 70:
            reasons.append("Overbought (RSI > 70)")
        else:
            reasons.append("RSI in neutral range")

        macd = self.technical_data.get('macd', 0.0)
        if macd > 0:
            reasons.append("Positive MACD (Bullish)")
        else:
            reasons.append("Negative MACD (Bearish)")

        self.recommendation = recommendation
        self.recommendation_reasons = reasons
        return recommendation

        """
        Calculate buy/add/sell recommendation based on final rating and key metrics.
        Returns:
            str: Recommendation (Buy, Add, Sell)
        """
        recommendation = "Hold"
        reasons = []

        if self.final_rating >= 80:
            recommendation = "Buy"
            reasons.append("High rating (>= 80)")
        elif 60 <= self.final_rating < 80:
            recommendation = "Add"
            reasons.append("Moderate rating (60-79)")
        elif self.final_rating < 60:
            recommendation = "Sell"
            reasons.append("Low rating (< 60)")

        rs = self.technical_data.get('rs', 50.0)
        if rs > 70:
            reasons.append("Strong RS > 70")
        elif rs < 30:
            recommendation = "Sell"
            reasons.append("Weak RS < 30")

        valuation_score = self.calculate_final_rating()[0][8]
        if valuation_score > 80:
            reasons.append("Attractive valuation (Valuation Score > 80)")
        elif valuation_score < 40:
            recommendation = "Sell"
            reasons.append("Overvalued (Valuation Score < 40)")

        rsi = self.technical_data.get('rsi', 50.0)
        current_price = self.technical_data.get('current_price', 1000.0)
        dma_value = self.technical_data.get('dma_value', 1000.0)
        current_volume = self.technical_data.get('current_volume', 1e6)
        vol_avg = self.technical_data.get('vol_avg', 1e6)

        try:
            data = yf.Ticker(self.ticker).history(period="60d")
            dma_20 = data['Close'].rolling(window=20).mean()[-1]
            dma_50 = data['Close'].rolling(window=50).mean()[-1]
        except Exception:
            dma_20 = dma_50 = current_price

        golden_crossover = current_price > dma_value
        high_volume_breakout = current_volume > vol_avg
        short_term_dma_crossover = dma_20 > dma_50

        if rsi < 30 and golden_crossover and high_volume_breakout and short_term_dma_crossover:
            recommendation = "Buy"
            reasons.append("Oversold (RSI < 30), Golden Crossover, High Volume Breakout, 20-DMA > 50-DMA")
        elif rsi < 30 and golden_crossover and high_volume_breakout:
            recommendation = "Buy"
            reasons.append("Oversold (RSI < 30), Golden Crossover, High Volume Breakout")
        elif rsi < 30:
            recommendation = "Buy"
            reasons.append("Oversold (RSI < 30)")
        elif rsi > 70:
            reasons.append("Overbought (RSI > 70)")
        else:
            reasons.append(f"RSI in neutral range (RSI={rsi:.2f})")

        macd = self.technical_data.get('macd', 0.0)
        if macd > 0:
            reasons.append("Positive MACD (Bullish)")
        else:
            reasons.append("Negative MACD (Bearish)")

        self.recommendation = recommendation
        self.recommendation_reasons = reasons
        return recommendation

    def get_results(self):
        """
        Return results as a dictionary for output.
        Returns:
            dict: Results including ticker, scores, final rating, and recommendation.
        """
        self.calculate_recommendation()
        return {
            'Ticker': self.ticker,
            'Final Rating (0-100)': self.final_rating,
            'Recommendation': self.recommendation,
            'Recommendation Reasons': '; '.join(self.recommendation_reasons),
            'Promoter Holding': f"{self.user_inputs.get('promoter_holding', 50.0)}% -> {min(100, self.user_inputs.get('promoter_holding', 50.0) * 2)}",
            'Inst. Holding': f"{self.user_inputs.get('inst_holding', 15.0)}% -> {min(100, self.user_inputs.get('inst_holding', 15.0) * 4)}",
            'D/E Ratio': f"{self.fundamental_data.get('debtToEquity', 1.0)} -> {max(0, 100 - self.fundamental_data.get('debtToEquity', 1.0) * 50)}",
            'RoE': f"{self.fundamental_data.get('returnOnEquity', 0.15)*100:.2f}% -> {min(100, self.fundamental_data.get('returnOnEquity', 0.15) * 500)}",
            'Valuation Score': self.calculate_final_rating()[0][8],
            'Profit CAGR 5Y': f"{self.user_inputs.get('cagr', 12.0)}% -> {min(100, max(0, self.user_inputs.get('cagr', 12.0) * 5))}",
            'Dividend Yield': f"{self.fundamental_data.get('dividendYield', 0.01)*100:.2f}% -> {min(100, self.fundamental_data.get('dividendYield', 0.01) * 5000)}",
            'Economic Moat': self.moat_basis,
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
        self.calculate_recommendation()
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
            ["Final Rating", "", final_rating],
            ["Recommendation", self.recommendation, ""],
            ["Recommendation Reasons", "; ".join(self.recommendation_reasons), ""]
        ]
        print(f"\nAnalysis for {self.ticker} ({self.exchange}):")
        print(tabulate(table, headers=["Parameter", "Value", "Score"], tablefmt="grid"))

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
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            
            data = yf.Ticker(self.ticker).history(period=f"{self.params['dma_length']}d")
            if data.empty:
                raise ValueError("No data for plotting")

            plt.figure(figsize=(10, 6))
            plt.plot(data.index, data['Close'], label='Close Price')
            plt.plot(data.index, data['Close'].rolling(window=self.params['dma_length']).mean(), label='200-Day DMA')
            plt.title(f"{self.ticker} Price and 200-Day DMA")
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
def fetch_screener_data(nse_code):
    """
    Fetch Promoter Holding, FII, DII, Profit Growth YoY, and Profit CAGR 5Y from Screener.in.
    Args:
        nse_code (str): NSE ticker without .NS suffix (e.g., 'CHOLAFIN')
    Returns:
        dict: Dictionary with fetched data or empty if failed.
    """
    data = {}
    try:
        # Normalize ticker: uppercase, replace spaces/hyphens
        nse_code = nse_code.upper().replace(' ', '-').replace('_', '-')
        url = f"https://www.screener.in/company/{nse_code}/consolidated/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Shareholding Pattern
        shareholding_section = soup.find("section", id="shareholding")
        if shareholding_section:
            tables = shareholding_section.find_all("table", class_="data-table")
            for table in tables:
                rows = table.find_all("tr")
                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) < 2:
                        continue
                    label = cells[0].text.strip().lower()
                    for cell in reversed(cells[1:]):
                        value = cell.text.strip().replace("%", "").replace(",", "")
                        try:
                            value = float(value)
                            if value < 0 or value > 100:
                                continue
                            if "promoter" in label:
                                data['promoter_holding'] = value
                            elif "fii" in label or "foreign" in label:
                                data['fii'] = value
                            elif "dii" in label or "domestic" in label:
                                data['dii'] = value
                            break
                        except ValueError:
                            for fallback_cell in cells[1:]:
                                value = fallback_cell.text.strip().replace("%", "").replace(",", "")
                                try:
                                    value = float(value)
                                    if value < 0 or value > 100:
                                        continue
                                    if "promoter" in label and 'promoter_holding' not in data:
                                        data['promoter_holding'] = value
                                    elif "fii" in label and 'fii' not in data:
                                        data['fii'] = value
                                    elif "dii" in label and 'dii' not in data:
                                        data['dii'] = value
                                    break
                                except ValueError:
                                    continue
            if 'fii' in data and 'dii' in data:
                data['inst_holding'] = min(100.0, data['fii'] + data['dii'])
            elif 'fii' in data:
                data['inst_holding'] = data['fii']
            elif 'dii' in data:
                data['inst_holding'] = data['dii']
            if 'promoter_holding' not in data:
                data['promoter_holding'] = 0.0
                logger.info(f"No promoter holding found for {nse_code}. Setting to 0%.")

        # Profit Growth YoY and Profit CAGR 5Y
        ratios_section = soup.find("section", id="top-ratios")
        if ratios_section:
            items = ratios_section.find_all("li", class_=["flex-row", "flex flex-space-between"])
            for item in items:
                text = item.text.lower().replace('\n', ' ').strip()
                value_span = item.find("span", class_=["number", "value"])
                if not value_span:
                    continue
                value = value_span.text.strip().replace("%", "").replace(",", "")
                try:
                    value = float(value)
                    if "profit growth" in text and ("1y" in text or "1 year" in text or "yoy" in text):
                        if -100 <= value <= 100:
                            data['profit_growth'] = value
                    elif "profit cagr" in text and ("5y" in text or "5 years" in text or "5-yr" in text):
                        if -100 <= value <= 100:
                            data['cagr'] = value
                except ValueError:
                    continue

        # Fallback: Check profit/loss table in financials
        if 'profit_growth' not in data or 'cagr' not in data:
            financials_section = soup.find("section", id="profit-loss")
            if financials_section:
                table = financials_section.find("table", class_="data-table")
                if table:
                    rows = table.find_all("tr")
                    for row in rows:
                        cells = row.find_all("td")
                        if len(cells) < 2:
                            continue
                        label = cells[0].text.strip().lower()
                        if "net profit" in label and 'profit_growth' not in data:
                            for i, cell in enumerate(reversed(cells[1:])):
                                value = cell.text.strip().replace("%", "").replace(",", "")
                                try:
                                    value = float(value)
                                    if i == 0 and -100 <= value <= 100:  # Latest YoY growth
                                        data['profit_growth'] = value
                                    break
                                except ValueError:
                                    continue
                        if "cagr" in label and 'cagr' not in data:
                            for cell in reversed(cells[1:]):
                                value = cell.text.strip().replace("%", "").replace(",", "")
                                try:
                                    value = float(value)
                                    if -100 <= value <= 100:
                                        data['cagr'] = value
                                    break
                                except ValueError:
                                    continue

        if not data:
            logger.warning(f"No relevant data found on Screener.in for {nse_code}")
        else:
            logger.info(f"Fetched Screener.in data for {nse_code}: {data}")
        time.sleep(2)  # Respect rate limits
        return data
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching Screener.in data for {nse_code}: {e}")
        return {}
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching Screener.in data for {nse_code}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Failed to fetch Screener.in data for {nse_code}: {e}")
        return {}

def fetch_nse_data(nse_code):
    """
    Fetch Promoter Holding, FII, DII from NSE website using equity-master and fallback endpoints.
    Args:
        nse_code (str): NSE ticker without .NS suffix (e.g., 'CHOLAFIN')
    Returns:
        dict: Dictionary with fetched data or empty if failed.
    """
    data = {}
    try:
        # Normalize ticker: uppercase, URL encode
        nse_code = urllib.parse.quote(nse_code.upper().replace(' ', '-'))
        # Try primary endpoint: equity-master
        url = f"https://www.nseindia.com/api/equity-master?symbol={nse_code}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": f"https://www.nseindia.com/get-quotes/equity?symbol={nse_code}",
            "X-Requested-With": "XMLHttpRequest",
            "Connection": "keep-alive"
        }
        session = requests.Session()
        session.get("https://www.nseindia.com/", headers=headers, timeout=10)
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        json_data = response.json()

        # Check for shareholding data
        if 'corpInfo' in json_data and 'shareHoldingPattern' in json_data['corpInfo']:
            quarters = json_data['corpInfo']['shareHoldingPattern'].get('quarters', [])
            if quarters:
                latest_quarter = quarters[-1]
                for category in latest_quarter.get('data', []):
                    label = category.get('category', '').lower()
                    percentage = category.get('percent', 0.0)
                    try:
                        percentage = float(percentage)
                        if percentage < 0 or percentage > 100:
                            continue
                        if 'promoter' in label:
                            data['promoter_holding'] = percentage
                        elif 'fii' in label or 'foreign' in label:
                            data['fii'] = percentage
                        elif 'dii' in label or 'domestic' in label:
                            data['dii'] = percentage
                    except (ValueError, TypeError):
                        continue
                if 'fii' in data and 'dii' in data:
                    data['inst_holding'] = min(100.0, data['fii'] + data['dii'])
                elif 'fii' in data:
                    data['inst_holding'] = data['fii']
                elif 'dii' in data:
                    data['inst_holding'] = data['dii']
                if 'promoter_holding' not in data:
                    data['promoter_holding'] = 0.0
                    logger.info(f"No promoter holding found for {nse_code}. Setting to 0%.")

        if data:
            logger.info(f"Fetched NSE data for {nse_code}: {data}")
            time.sleep(2)
            return data

        # Fallback: company-profile endpoint
        logger.info(f"No shareholding data in equity-master for {nse_code}. Trying company-profile...")
        url = f"https://www.nseindia.com/api/company-profile?symbol={nse_code}"
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        json_data = response.json()

        if 'shareHoldingPattern' in json_data:
            quarters = json_data['shareHoldingPattern'].get('quarters', [])
            if quarters:
                latest_quarter = quarters[-1]
                for category in latest_quarter.get('data', []):
                    label = category.get('category', '').lower()
                    percentage = category.get('percent', 0.0)
                    try:
                        percentage = float(percentage)
                        if percentage < 0 or percentage > 100:
                            continue
                        if 'promoter' in label:
                            data['promoter_holding'] = percentage
                        elif 'fii' in label or 'foreign' in label:
                            data['fii'] = percentage
                        elif 'dii' in label or 'domestic' in label:
                            data['dii'] = percentage
                    except (ValueError, TypeError):
                        continue
                if 'fii' in data and 'dii' in data:
                    data['inst_holding'] = min(100.0, data['fii'] + data['dii'])
                elif 'fii' in data:
                    data['inst_holding'] = data['fii']
                elif 'dii' in data:
                    data['inst_holding'] = data['dii']
                if 'promoter_holding' not in data:
                    data['promoter_holding'] = 0.0
                    logger.info(f"No promoter holding found for {nse_code}. Setting to 0%.")

        if data:
            logger.info(f"Fetched NSE data for {nse_code}: {data}")
        else:
            logger.warning(f"No relevant shareholding data found on NSE for {nse_code}")
        time.sleep(2)  # Respect rate limits
        return data
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            logger.error(f"Access denied (403) fetching NSE data for {nse_code}. Check cookies/headers.")
        elif e.response.status_code == 404:
            logger.error(f"Ticker not found (404) for {nse_code}. Verify ticker or NSE listing.")
        else:
            logger.error(f"HTTP error fetching NSE data for {nse_code}: {e}")
        return {}
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching NSE data for {nse_code}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Failed to fetch NSE data for {nse_code}: {e}")
        return {}

    """
    Fetch Promoter Holding, FII, DII from NSE website using equity-master and fallback endpoints.
    Args:
        nse_code (str): NSE ticker without .NS suffix (e.g., 'CARTRADE')
    Returns:
        dict: Dictionary with fetched data or empty if failed.
    """
    data = {}
    try:
        # Normalize ticker: uppercase, URL encode spaces
        nse_code = urllib.parse.quote(nse_code.upper().replace(' ', '-'))
        # Try primary endpoint: equity-master
        url = f"https://www.nseindia.com/api/equity-master?symbol={nse_code}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": f"https://www.nseindia.com/get-quotes/equity?symbol={nse_code}",
            "X-Requested-With": "XMLHttpRequest",
            "Connection": "keep-alive"
        }
        session = requests.Session()
        # Set cookies
        session.get("https://www.nseindia.com/", headers=headers, timeout=10)
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        json_data = response.json()

        # Check for shareholding data
        if 'corpInfo' in json_data and 'shareHoldingPattern' in json_data['corpInfo']:
            quarters = json_data['corpInfo']['shareHoldingPattern'].get('quarters', [])
            if quarters:
                latest_quarter = quarters[-1]
                for category in latest_quarter.get('data', []):
                    label = category.get('category', '').lower()
                    percentage = category.get('percent', 0.0)
                    try:
                        percentage = float(percentage)
                        if percentage < 0 or percentage > 100:
                            continue
                        if 'promoter' in label:
                            data['promoter_holding'] = percentage
                        elif 'fii' in label or 'foreign' in label:
                            data['fii'] = percentage
                        elif 'dii' in label or 'domestic' in label:
                            data['dii'] = percentage
                    except (ValueError, TypeError):
                        continue
                if 'fii' in data and 'dii' in data:
                    data['inst_holding'] = min(100.0, data['fii'] + data['dii'])
                elif 'fii' in data:
                    data['inst_holding'] = data['fii']
                elif 'dii' in data:
                    data['inst_holding'] = data['dii']
                if 'promoter_holding' not in data:
                    data['promoter_holding'] = 0.0
                    logger.info(f"No promoter holding found for {nse_code}. Setting to 0%.")

        if data:
            logger.info(f"Fetched NSE data for {nse_code}: {data}")
            time.sleep(2)
            return data

        # Fallback: try company-profile endpoint
        logger.info(f"No shareholding data in equity-master for {nse_code}. Trying company-profile...")
        url = f"https://www.nseindia.com/api/company-profile?symbol={nse_code}"
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        json_data = response.json()

        if 'shareHoldingPattern' in json_data:
            quarters = json_data['shareHoldingPattern'].get('quarters', [])
            if quarters:
                latest_quarter = quarters[-1]
                for category in latest_quarter.get('data', []):
                    label = category.get('category', '').lower()
                    percentage = category.get('percent', 0.0)
                    try:
                        percentage = float(percentage)
                        if percentage < 0 or percentage > 100:
                            continue
                        if 'promoter' in label:
                            data['promoter_holding'] = percentage
                        elif 'fii' in label or 'foreign' in label:
                            data['fii'] = percentage
                        elif 'dii' in label or 'domestic' in label:
                            data['dii'] = percentage
                    except (ValueError, TypeError):
                        continue
                if 'fii' in data and 'dii' in data:
                    data['inst_holding'] = min(100.0, data['fii'] + data['dii'])
                elif 'fii' in data:
                    data['inst_holding'] = data['fii']
                elif 'dii' in data:
                    data['inst_holding'] = data['dii']
                if 'promoter_holding' not in data:
                    data['promoter_holding'] = 0.0
                    logger.info(f"No promoter holding found for {nse_code}. Setting to 0%.")

        if data:
            logger.info(f"Fetched NSE data for {nse_code}: {data}")
        else:
            logger.warning(f"No relevant shareholding data found on NSE for {nse_code}")
        time.sleep(2)  # Respect rate limits
        return data
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            logger.error(f"Access denied (403) fetching NSE data for {nse_code}. Check cookies/headers.")
        elif e.response.status_code == 404:
            logger.error(f"Ticker not found (404) for {nse_code}. Verify ticker or NSE listing.")
        else:
            logger.error(f"HTTP error fetching NSE data for {nse_code}: {e}")
        return {}
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching NSE data for {nse_code}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Failed to fetch NSE data for {nse_code}: {e}")
        return {}

    """
    Fetch Promoter Holding, FII, DII from NSE website using company profile API.
    Args:
        nse_code (str): NSE ticker without .NS suffix (e.g., 'HDFCBANK')
    Returns:
        dict: Dictionary with fetched data or empty if failed.
    """
    data = {}
    try:
        # Normalize ticker
        nse_code = nse_code.upper().replace(' ', '%20')
        url = f"https://www.nseindia.com/api/company-profile?symbol={nse_code}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.nseindia.com/",
            "Connection": "keep-alive"
        }
        session = requests.Session()
        # Set cookies by visiting homepage
        session.get("https://www.nseindia.com/", headers=headers, timeout=10)
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        json_data = response.json()

        # Extract shareholding data from 'shareHoldingPattern'
        if 'shareHoldingPattern' in json_data:
            quarters = json_data['shareHoldingPattern'].get('quarters', [])
            if quarters:
                latest_quarter = quarters[-1]  # Latest quarter
                for category in latest_quarter.get('data', []):
                    label = category.get('category', '').lower()
                    percentage = category.get('percent', 0.0)
                    try:
                        percentage = float(percentage)
                        if percentage < 0 or percentage > 100:
                            continue
                        if 'promoter' in label:
                            data['promoter_holding'] = percentage
                        elif 'fii' in label or 'foreign' in label:
                            data['fii'] = percentage
                        elif 'dii' in label or 'domestic' in label:
                            data['dii'] = percentage
                    except (ValueError, TypeError):
                        continue
            # Calculate Institutional Holding
            if 'fii' in data and 'dii' in data:
                data['inst_holding'] = min(100.0, data['fii'] + data['dii'])
            elif 'fii' in data:
                data['inst_holding'] = data['fii']
            elif 'dii' in data:
                data['inst_holding'] = data['dii']

        if data:
            logger.info(f"Fetched NSE data for {nse_code}: {data}")
        else:
            logger.warning(f"No relevant shareholding data found on NSE for {nse_code}")
        time.sleep(2)  # Respect rate limits
        return data
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            logger.error(f"Access denied (403) fetching NSE data for {nse_code}. Check cookies/headers.")
        else:
            logger.error(f"HTTP error fetching NSE data for {nse_code}: {e}")
        return {}
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching NSE data for {nse_code}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Failed to fetch NSE data for {nse_code}: {e}")
        return {}

def process_portfolio(input_excel, output_excel="portfolio_rankings.xlsx"):
    """
    Process a portfolio from an Excel file and generate an output Excel with rankings.
    Args:
        input_excel (str): Path to input Excel file.
        output_excel (str): Path to output Excel file.
    """
    df = pd.read_excel(input_excel)
    required_columns = ['Ticker']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Input Excel must contain 'Ticker' column")

    if df['Ticker'].duplicated().any():
        duplicates = df[df['Ticker'].duplicated()]['Ticker'].tolist()
        logger.warning(f"Duplicate tickers found: {', '.join(duplicates)}. Processing first occurrence only.")
        df = df.drop_duplicates(subset=['Ticker'], keep='first')

    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    results = []
    all_calculation_details = []
    missing_inputs = []
    data_cache = {}  # Cache fetched data to avoid repeated requests

    for _, row in df.iterrows():
        ticker = str(row['Ticker']).strip().upper()
        exchange = row.get('Exchange', 'NSE').strip().upper() if 'Exchange' in df.columns else 'NSE'

        if exchange not in SUPPORTED_EXCHANGES:
            logger.error(f"Unsupported exchange for {ticker}: {exchange}. Supported exchanges: {', '.join(SUPPORTED_EXCHANGES)}")
            results.append({
                'Ticker': ticker,
                'Final Rating (0-100)': 'Error',
                'Calculation Details': f"Error: Unsupported exchange {exchange}"
            })
            continue

        suffix = config[exchange].get('suffix', '')
        if suffix and not ticker.endswith(f".{suffix}"):
            ticker += suffix

        print(f"\nProcessing {ticker} ({exchange})...")
        try:
            analyzer = StockAnalyzer(ticker, exchange=exchange)
            analyzer.fetch_fundamental_data()

            inputs_dict = {
                'promoter_holding': row.get('Promoter Holding (%)', np.nan),
                'inst_holding': row.get('Inst. Holding (%)', np.nan),
                'profit_growth': row.get('Profit Growth YoY (%)', np.nan),
                'cagr': row.get('Profit CAGR 5Y (%)', np.nan),
                'moat': row.get('Economic Moat', None)
            }

            # Fetch missing inputs for NSE stocks
            if exchange == 'NSE':
                nse_code = ticker.replace(".NS", "")
                fetched_data = None

                # Check cache first
                if nse_code in data_cache:
                    fetched_data = data_cache[nse_code]
                    logger.info(f"Using cached data for {nse_code}: {fetched_data}")
                else:
                    # Try Screener.in
                    fetched_data = fetch_screener_data(nse_code)
                    if not all(k in fetched_data for k in ['promoter_holding', 'inst_holding', 'profit_growth', 'cagr']):
                        logger.info(f"Some data missing from Screener.in for {nse_code}. Trying NSE...")
                        nse_data = fetch_nse_data(nse_code)
                        # Merge data, prioritizing Screener.in for profit metrics
                        fetched_data.update(nse_data)
                    if fetched_data:
                        data_cache[nse_code] = fetched_data

                # Update inputs_dict with fetched data if missing
                for key in ['promoter_holding', 'inst_holding', 'profit_growth', 'cagr']:
                    if pd.isna(inputs_dict[key]) and key in fetched_data and fetched_data[key] is not None:
                        value = fetched_data[key]
                        if key in ['promoter_holding', 'inst_holding'] and (value < 0 or value > 100):
                            logger.warning(f"Invalid {key} value for {ticker}: {value}. Skipping.")
                            continue
                        if key in ['profit_growth', 'cagr'] and (value < -100 or value > 100):
                            logger.warning(f"Invalid {key} value for {ticker}: {value}. Skipping.")
                            continue
                        inputs_dict[key] = value
                        analyzer.calculation_details.append(f"Fetched {key} for {ticker} from {'Screener.in' if key in fetched_data else 'NSE'}: {value}")

            missing = [key for key, value in inputs_dict.items() if pd.isna(value) and key != 'moat']
            if missing:
                missing_inputs.append(f"{ticker}: Missing {', '.join(missing)}")

            analyzer.set_user_inputs(inputs_dict)
            analyzer.calculate_technical_metrics()
            analyzer.calculate_final_rating()
            result = analyzer.get_results()
            results.append(result)
            all_calculation_details.extend([f"{ticker}: {detail}" for detail in analyzer.calculation_details])
            analyzer.plot_metrics(plot_dir)

        except Exception as e:
            logger.error(f"Failed to process {ticker}: {e}")
            results.append({
                'Ticker': ticker,
                'Final Rating (0-100)': 'Error',
                'Calculation Details': f"Error: {str(e)}. Check ticker validity or try again later."
            })
            all_calculation_details.append(f"{ticker}: Error: {str(e)}")

    if missing_inputs:
        print("\nWarning: Missing inputs in Excel file (defaults used):")
        for entry in missing_inputs:
            print(f"- {entry}")

    results_df = pd.DataFrame(results)
    calc_details_df = pd.DataFrame(all_calculation_details, columns=['Calculation Details'])

    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Rankings', index=False)
        calc_details_df.to_excel(writer, sheet_name='Calculation Details', index=False)
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

def main():
    """Main function to run stock or portfolio analysis."""
    parser = argparse.ArgumentParser(description="Stock Ranking Analysis Tool")
    parser.add_argument('--portfolio', type=str, help="Path to portfolio Excel file (e.g., portfolio.xlsx)")
    args = parser.parse_args()

    print("Disclaimer: This tool is for educational purposes only and not financial advice. \nThe developer is not liable for any losses or actions by SEBI or other regulators. \nUsers are responsible for their investment decisions and should consult a registered advisor.\n")

    if args.portfolio:
        if not os.path.exists(args.portfolio):
            print(f"Error: Portfolio file {args.portfolio} not found.")
            return
        process_portfolio(args.portfolio)
    else:
        session = PromptSession(multiline=False)
        print(f"Supported exchanges: {', '.join(SUPPORTED_EXCHANGES)}")
        ticker = session.prompt("Enter the stock ticker (e.g., RELIANCE.NS for NSE, NXPI for NASDAQ, AAPL for NYSE): ").strip().upper()
        exchange = session.prompt(f"Enter the exchange (default NSE): ", default="NSE").strip().upper()

        if exchange not in SUPPORTED_EXCHANGES:
            print(f"Error: Unsupported exchange {exchange}. Supported exchanges: {', '.join(SUPPORTED_EXCHANGES)}")
            return

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
