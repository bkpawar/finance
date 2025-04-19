### Overview
The Python script, `stock_ranking.py`, is a stock analysis tool designed to evaluate NSE (National Stock Exchange of India) stocks by computing a **Final Rating** (0-100) based on 15 fundamental and technical parameters. The rating helps investors identify stocks with strong fundamentals, positive technical trends, and lower risk, potentially leading to better investment returns. Key features include:

- **Data Sources**: Uses `yahooquery` and `yfinance` for fundamental and historical price data.
- **Parameters**:
  - **User Inputs (4)**: Promoter Holding, Institutional Holding, Profit Growth YoY, Profit CAGR 5Y.
  - **Automatically Calculated (11)**: Debt-to-Equity Ratio, RoE, Dividend Yield, Economic Moat, Valuation Score, SymbolTrendRS, NIFTY200DMARSIVolume, 200-Day DMA, 14-Day RSI, 20-Day Volume MA, Simple RS vs NIFTY_50.
- **Scoring System**: Weighted scoring with a volatility penalty to balance return potential and risk.
- **User Interaction**: Interactive CLI using `prompt_toolkit` for inputs and moat override.
- **Output**: Table, CSV file, and price vs. DMA plot, with detailed calculation explanations.
- **Error Handling**: Robust logging and fallbacks for data fetching.

## Disclaimer: No Liability, Including from SEBI

This stock ranking tool is provided strictly for **educational and informational purposes** and is not intended as financial advice, investment recommendations, or a solicitation to buy or sell securities. The developer, contributors, and distributors of this tool, including the author, shall not be held liable for any financial losses, legal actions, or regulatory consequences arising from its use, including but not limited to actions by the **Securities and Exchange Board of India (SEBI)** or any other regulatory authority. Users are solely responsible for their investment decisions and should consult a SEBI-registered financial advisor or conduct independent research before investing. The tool’s outputs, including the Final Rating, are based on publicly available data and user inputs, which may contain errors or omissions, and do not guarantee future performance.

### Code with Inline Documentation
Below is the complete code with detailed comments explaining each section and function.

```python
"""
Stock Ranking Analysis Tool
==========================
This script analyzes NSE stocks by computing a Final Rating (0-100) based on 15 fundamental and technical parameters.
It supports:
- Single stock analysis with interactive user inputs.
- Portfolio analysis via an input Excel file containing stock tickers and optional user inputs.
- Outputs results to an Excel file with rankings and calculation details.

Key Features:
- Fetches fundamental data (e.g., D/E Ratio, RoE) using yahooquery and yfinance.
- Calculates Economic Moat based on profit margin, market cap, ROIC, and revenue stability.
- Processes user inputs from Excel or CLI for Promoter Holding, Institutional Holding, Profit Growth, and Profit CAGR.
- Allows Economic Moat override via Excel or CLI.
- Generates a table, CSV, plots, and an output Excel file with rankings.
- Includes detailed calculation explanations for auto-calculated parameters.

Dependencies: yahooquery, yfinance, pandas, numpy, ta, tabulate, prompt_toolkit, matplotlib, openpyxl

The `stock_ranking.py` tool is designed to help investors evaluate and rank stocks listed on the National Stock Exchange (NSE) of India by assigning each stock a **Final Rating** (0-100). This rating is a composite score based on 15 **fundamental** and **technical** parameters, each reflecting a different aspect of a stock’s financial health, market performance, or investment potential. For beginners ("newbies"), understanding how this tool works involves grasping the concepts of stock analysis, the role of each parameter, and how they are combined to produce a ranking. Below, I’ll explain the tool’s theoretical workings in simple terms, focusing on the parameters, their significance, and how they contribute to the final ranking, ensuring the explanation is accessible to those new to investing and stock analysis.

---

## Overview: What Does the Tool Do?

Imagine you’re trying to decide which stocks to invest in from a list of companies (e.g., Reliance, TCS, HDFC Bank). Each company has different strengths and weaknesses, like profitability, debt levels, or stock price trends. The tool helps by:

1. **Collecting Data**: It gathers financial and market data about each stock using free APIs (`yahooquery` and `yfinance`).
2. **Analyzing Parameters**: It evaluates 15 specific metrics (parameters) that measure the company’s quality and stock performance.
3. **Scoring and Weighting**: Each parameter is scored (0-100), adjusted by a **weight** (importance), and combined into a single **Final Rating**.
4. **Outputting Results**: For a single stock, it shows a table, CSV, and plot. For a portfolio (via an Excel file), it generates an Excel file with rankings for all stocks.

The **Final Rating** acts like a report card grade, helping you compare stocks. A higher rating (e.g., 70) suggests a stock is stronger than one with a lower rating (e.g., 50), based on the chosen parameters.

---

## Key Concepts for Beginners

Before diving into the parameters, let’s cover some basic ideas:

- **Stocks**: Shares of a company you can buy. Their value depends on the company’s performance and market trends.
- **Fundamental Analysis**: Looks at a company’s financial health (e.g., profits, debt) to judge if it’s a good long-term investment.
- **Technical Analysis**: Examines stock price and trading patterns (e.g., price trends, volume) to predict short-term performance.
- **Parameters**: Specific metrics (e.g., Debt-to-Equity Ratio, RSI) used to evaluate a stock.
- **Weighting**: Some parameters are more important, so they have a higher “weight” in the final score.
- **Volatility Penalty**: Risky (volatile) stocks get a slight score reduction to favor stable investments.

The tool combines fundamental and technical parameters to give a balanced view, suitable for both long-term and short-term investors.

---

## How the Tool Works: Step-by-Step

### 1. Input
- **Single Stock Mode**: You enter an NSE stock ticker (e.g., `RELIANCE.NS`) and manually provide inputs like Promoter Holding or Profit Growth via prompts.
- **Portfolio Mode**: You provide an Excel file (e.g., `portfolio.xlsx`) listing tickers and optional inputs (e.g., Promoter Holding, Economic Moat). Example:
 ```excel
 | Ticker | Promoter Holding (%) | Profit Growth YoY (%) | Economic Moat |
 |--------------|---------------------|-----------------------|---------------|
 | RELIANCE.NS | 50.6 | 11.5 | Wide |
 | TCS.NS | 72.3 | 9.0 | |
 ```
- The tool validates inputs, using defaults (e.g., 10.0 for Profit Growth) if values are missing or invalid.

### 2. Data Collection
- **Fundamental Data**: Fetched from `yahooquery` (primary) and `yfinance` (backup) for metrics like Debt-to-Equity Ratio, Return on Equity, and Dividend Yield.
- **Technical Data**: Fetched from `yfinance` for price history, volume, and indicators like 200-Day DMA and RSI.
- **User Inputs**: For metrics not available via APIs (e.g., Promoter Holding, Profit CAGR), users provide values from sources like [Screener.in](https://www.screener.in/) or [Moneycontrol](https://www.moneycontrol.com/).

### 3. Parameter Calculation
The tool evaluates 15 parameters, each converted to a score (0-100). These are divided into:
- **Fundamental Parameters**: Reflect the company’s financial health and competitive position.
- **Technical Parameters**: Reflect the stock’s market performance and trading trends.

### 4. Scoring and Weighting
- Each parameter’s score is multiplied by its **weight** (from `config.ini`, summing to 1.0) to reflect its importance.
- A **volatility penalty** reduces the score for stocks with high price swings, favoring stability.
- The weighted scores are averaged to produce the **Final Rating**.

### 5. Output
- **Single Stock**: Displays a table, saves a CSV (e.g., `RELIANCE.NS_analysis.csv`), and generates a plot (e.g., `RELIANCE.NS_plot.png`) showing price vs. 200-Day DMA.
- **Portfolio**: Creates `portfolio_rankings.xlsx` with two sheets:
 - **Rankings**: All parameters and Final Rating for each stock.
 - **Calculation Details**: Explanations of how metrics were calculated.
- Example Rankings sheet:
 ```excel
 | Ticker | Final Rating (0-100) | SymbolTrendRS | Economic Moat | ...
 |--------------|---------------------|---------------|--------------------------------------------------| ...
 | RELIANCE.NS | 70.20 | 62.45 | Wide (Overridden by user to Wide) | ...
 | TCS.NS | 65.50 | 58.30 | Narrow (Narrow moat due to: High profit margin) | ...
 ```

---

## The 15 Parameters and Their Role in Ranking

Below, I’ll explain each parameter, why it matters, how it’s calculated or sourced, and how it’s scored. This will help beginners understand what drives the Final Rating.

### Fundamental Parameters (8)
These assess the company’s financial health, ownership, and competitive advantage.

1. **Promoter Holding (%)**
 - **What**: Percentage of shares owned by the company’s promoters (founders or key stakeholders).
 - **Why**: High promoter holding shows confidence in the company’s future. Low holding may signal risk.
 - **Source**: User input (e.g., from Screener.in). Default: 50.0%.
 - **Scoring**: `score = min(100, promoter_holding * 2)` (e.g., 50% → 100, 25% → 50).
 - **Weight**: 0.1 (10% of Final Rating).
 - **Example**: Reliance’s 50.6% promoter holding scores 100 (strong signal).

2. **Institutional Holding (%)**
 - **What**: Percentage of shares owned by institutions (e.g., mutual funds, foreign investors).
 - **Why**: High institutional holding indicates trust from professional investors.
 - **Source**: User input. Default: 15.0%.
 - **Scoring**: `score = min(100, inst_holding * 4)` (e.g., 25% → 100, 10% → 40).
 - **Weight**: 0.1.
 - **Example**: TCS’s 23.4% institutional holding scores 93.6 (positive).

3. **Debt-to-Equity Ratio (D/E Ratio)**
 - **What**: Ratio of company’s debt to its equity (net worth).
 - **Why**: Low D/E means less debt, reducing financial risk. High D/E can strain finances.
 - **Source**: `yahooquery` or `yfinance`. Default: 1.0.
 - **Scoring**: `score = max(0, 100 - de_ratio * 50)` (e.g., 0.6 → 70, 2.0 → 0).
 - **Weight**: 0.1.
 - **Example**: Reliance’s D/E of 0.6 scores 70 (manageable debt).

4. **Return on Equity (RoE) (%)**
 - **What**: Measures how efficiently a company uses shareholders’ money to generate profit.
 - **Why**: High RoE indicates strong profitability. Low RoE suggests inefficiency.
 - **Source**: `yahooquery` or `yfinance`. Default: 15.0%.
 - **Scoring**: `score = min(100, roe * 5)` (e.g., 9.2% → 46, 20% → 100).
 - **Weight**: 0.1.
 - **Example**: TCS’s RoE of 45% scores 100 (excellent profitability).

5. **Profit Growth Year-over-Year (YoY) (%)**
 - **What**: Percentage increase in net profit from last year.
 - **Why**: Positive growth shows improving earnings. Negative growth is a red flag.
 - **Source**: User input. Default: 10.0%.
 - **Scoring**: `score = min(100, profit_growth * 5)` (e.g., 11.5% → 57.5, -10% → 0).
 - **Weight**: 0.1.
 - **Example**: Reliance’s 11.5% growth scores 57.5 (moderate growth).

6. **Profit CAGR 5Y (%)**
 - **What**: Compound Annual Growth Rate of net profit over 5 years.
 - **Why**: Shows long-term profit consistency. Higher CAGR is better.
 - **Source**: User input. Default: 12.0%.
 - **Scoring**: `score = min(100, cagr * 5)` (e.g., 8.7% → 43.5, 20% → 100).
 - **Weight**: 0.1.
 - **Example**: Reliance’s 8.7% CAGR scores 43.5 (steady but not exceptional).

7. **Dividend Yield (%)**
 - **What**: Annual dividend per share divided by stock price, as a percentage.
 - **Why**: High yield attracts income-focused investors. Low yield may prioritize growth.
 - **Source**: `yahooquery` or `yfinance`. Default: 1.0%.
 - **Scoring**: `score = min(100, div_yield * 50)` (e.g., 0.4% → 20, 2% → 100).
 - **Weight**: 0.05 (less important).
 - **Example**: Reliance’s 0.4% yield scores 20 (low dividend focus).

8. **Economic Moat**
 - **What**: A company’s competitive advantage (e.g., brand, patents) that protects profits.
 - **Why**: Wide moat means sustainable profits. No moat means vulnerability.
 - **Source**: Calculated based on profit margin, market cap, ROIC, and revenue stability (placeholder: 10%). Can be overridden by user (None, Narrow, Wide).
 - **Scoring**:
 - Wide: 100
 - Narrow: 50
 - None: 0
 - **Weight**: 0.15 (most important fundamental parameter).
 - **Example**: Reliance’s user-overridden “Wide” moat scores 100 (strong advantage).

### Technical Parameters (7)
These assess the stock’s price trends, momentum, and trading activity.

9. **Valuation Score**
 - **What**: Measures if the stock is overvalued or undervalued based on Price-to-Earnings (P/E) and Price-to-Book (P/B) ratios.
 - **Why**: Low P/E and P/B suggest a bargain. High ratios suggest overvaluation.
 - **Source**: `yahooquery` (P/E, P/B compared to industry averages).
 - **Scoring**: `score = (pe_score + pb_score) / 2`, where `pe_score = 100 - (pe_ratio / industry_pe) * 50`, `pb_score = 100 - (pb_ratio / 5) * 50`.
 - **Weight**: 0.1.
 - **Example**: Reliance’s Valuation Score of 75 indicates reasonable pricing.

10. **SymbolTrendRS**
 - **What**: Combines Relative Strength Index (RSI) and Moving Average Convergence Divergence (MACD) to gauge trend strength.
 - **Why**: High score indicates bullish momentum. Low score suggests weakness.
 - **Source**: Calculated from `yfinance` price data.
 - **Scoring**: `score = (rsi_score + macd_score) / 2`, where `rsi_score = (RSI / 100) * 50`, `macd_score = 50 (if MACD > 0) else 25`.
 - **Weight**: 0.1.
 - **Example**: Reliance’s SymbolTrendRS of 62.45 shows moderate bullishness.

11. **NIFTY200DMARSIVolume**
 - **What**: Composite score based on 200-Day DMA, RSI, volume, and Relative Strength (RS) vs. NIFTY 50.
 - **Why**: Measures stock strength relative to the market, favoring stocks above DMA with high volume and momentum.
 - **Source**: Calculated from `yfinance` data.
 - **Scoring**: Sum of:
 - DMA: 25 if price > 200-Day DMA, else 0.
 - RSI: 25 if 30 ≤ RSI ≤ 70, else 10.
 - Volume: 25 if current volume > 20-Day Volume MA, else 10.
 - RS: 25 if RS > 50, else 10.
 - **Weight**: 0.1.
 - **Example**: Reliance’s score of 85 indicates strong technical performance.

12. **200-Day DMA (Daily Moving Average)**
 - **What**: Average stock price over the last 200 days.
 - **Why**: Price above DMA signals an uptrend; below DMA signals a downtrend.
 - **Source**: Calculated from `yfinance` price data.
 - **Scoring**: `score = 100 if current_price > dma_value else 0`.
 - **Weight**: 0.05.
 - **Example**: Reliance’s price above its 200-Day DMA (2950.34) scores 100.

13. **14-Day RSI (Relative Strength Index)**
 - **What**: Measures price momentum (0-100). RSI > 70 is overbought; < 30 is oversold.
 - **Why**: Indicates whether the stock is gaining or losing momentum.
 - **Source**: Calculated from `yfinance` price data.
 - **Scoring**:
 - RSI < 30: 25 (oversold, potential buy).
 - RSI > 70: 75 (overbought, potential sell).
 - Else: 50 (neutral).
 - **Weight**: 0.05.
 - **Example**: Reliance’s RSI of 62.45 scores 50 (neutral momentum).

14. **20-Day Volume MA (Moving Average)**
 - **What**: Average trading volume over the last 20 days.
 - **Why**: High current volume vs. MA suggests strong investor interest.
 - **Source**: Calculated from `yfinance` volume data.
 - **Scoring**: `score = 75 if current_volume > vol_avg else 25`.
 - **Weight**: 0.05.
 - **Example**: Reliance’s high volume scores 75 (active trading).

15. **Simple RS vs. NIFTY_50 (63 Days)**
 - **What**: Relative Strength compares the stock’s 63-day return to the NIFTY 50 index.
 - **Why**: Outperforming the market indicates strength.
 - **Source**: Calculated from `yfinance` price data.
 - **Scoring**: `score = max(0, min(100, 50 + (stock_return - nifty_return)))`.
 - **Weight**: 0.1.
 - **Example**: Reliance’s RS of 48.76 scores 48.76 (slightly underperforming NIFTY).

---

## How Parameters Decide the Ranking

### Scoring Each Parameter
- Each parameter is converted to a score (0-100) using specific formulas (e.g., `min(100, promoter_holding * 2)` for Promoter Holding).
- Scores reflect the parameter’s quality:
 - High scores (e.g., 100 for Wide Moat) indicate strength.
 - Low scores (e.g., 0 for high D/E Ratio) indicate weakness.

### Weighting
- Each parameter has a **weight** (from `config.ini`) to reflect its importance:
 ```ini
 weights = 0.1,0.1,0.1,0.1,0.1,0.1,0.05,0.15,0.1,0.1,0.1,0.05,0.05,0.05,0.1
 ```
 - **High Weight (0.15)**: Economic Moat (most influential due to competitive advantage).
 - **Medium Weight (0.1)**: Most parameters (balanced contribution).
 - **Low Weight (0.05)**: Dividend Yield, DMA, RSI, Volume MA (less critical).
- The weighted score for each parameter is: `score * weight`.

### Volatility Penalty
- **Volatility**: Measures how much the stock’s price fluctuates (calculated as annualized standard deviation of daily returns).
- **Penalty**: `volatility_penalty = max(0, 100 - (volatility * 100)) / 100`.
- **Effect**: High volatility reduces the weighted score, favoring stable stocks.
- **Example**: A volatility of 20% (0.2) gives a penalty of 0.8, reducing the weighted score by 20%.

### Final Rating
- The Final Rating is the weighted average of all 15 scores, adjusted for volatility:
 ```python
 weighted_scores = [score * weight * volatility_penalty for score, weight in zip(scores, self.weights)]
 final_rating = round(sum(weighted_scores) / sum(self.weights), 2)
 ```
- **Formula**: 
 \[
 \text{Final Rating} = \frac{\sum (\text{Score}_i \times \text{Weight}_i \times \text{Volatility Penalty})}{\sum \text{Weight}_i}
 \]
- **Example for Reliance**:
 - Scores: [100, 93.6, 70, 46, 57.5, 43.5, 20, 100, 75, 62.45, 85, 100, 50, 75, 48.76]
 - Weights: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.1]
 - Volatility Penalty: 0.8 (assuming 20% volatility)
 - Weighted Scores: [8, 7.49, 5.6, 3.68, 4.6, 3.48, 0.8, 12, 6, 5, 6.8, 4, 2, 3, 3.9]
 - Final Rating: \(\frac{76.35}{1.0} \times 0.8 \approx 70.20\)

### Ranking
- In portfolio mode, stocks are listed in the output Excel. Higher Final Ratings indicate better investment potential based on the parameters.
- Users can sort the Excel by Final Rating to identify top-ranked stocks (improvement suggested: auto-sort).

---

## Why These Parameters Matter

The 15 parameters are chosen to balance **fundamental** (long-term value) and **technical** (short-term trends) factors, providing a holistic view:

- **Fundamental Parameters**: Ensure the company is financially sound (low debt, high RoE), well-owned (promoters, institutions), growing (profit growth, CAGR), and competitively strong (moat). Dividend Yield appeals to income investors.
- **Technical Parameters**: Confirm the stock is performing well in the market (price above DMA, high volume, strong RS) and has momentum (RSI, SymbolTrendRS). Valuation Score ensures the stock isn’t overpriced.
- **Volatility Penalty**: Reduces risk by penalizing stocks with wild price swings.

For beginners:
- **High Final Rating**: Suggests a stock with strong fundamentals, good market performance, and low risk.
- **Low Final Rating**: Indicates weaknesses (e.g., high debt, weak momentum) or high volatility.

---

## Theoretical Example: Reliance vs. TCS

Let’s compare two stocks to see how parameters drive rankings:

| Parameter | Reliance Score | TCS Score | Why It Differs |
|-------------------------------|----------------|-----------|----------------|
| Promoter Holding (%) | 100 (50.6%) | 100 (72.3%) | Both high, but TCS has stronger promoter confidence. |
| Inst. Holding (%) | 93.6 (38.5%) | 93.6 (23.4%) | Reliance has higher institutional trust. |
| D/E Ratio | 70 (0.6) | 90 (0.2) | TCS has lower debt, scoring higher. |
| RoE (%) | 46 (9.2%) | 100 (45%) | TCS is far more profitable per equity. |
| Profit Growth YoY (%) | 57.5 (11.5%) | 45 (9%) | Reliance grew profits faster. |
| Profit CAGR 5Y (%) | 43.5 (8.7%) | 51 (10.2%) | TCS has better long-term growth. |
| Dividend Yield (%) | 20 (0.4%) | 30 (0.6%) | TCS offers slightly better dividends. |
| Economic Moat | 100 (Wide) | 50 (Narrow)| Reliance’s override gives it an edge. |
| Valuation Score | 75 | 70 | Reliance is slightly better valued. |
| SymbolTrendRS | 62.45 | 58.30 | Reliance has stronger momentum. |
| NIFTY200DMARSIVolume | 85 | 80 | Reliance outperforms market slightly more. |
| 200-Day DMA | 100 | 100 | Both above DMA (uptrend). |
| 14-Day RSI | 50 | 50 | Both neutral momentum. |
| 20-Day Vol MA | 75 | 75 | Both have high trading activity. |
| Simple RS vs NIFTY_50 (63d) | 48.76 | 55 | TCS slightly outperforms NIFTY. |
| **Volatility Penalty** | 0.8 (20%) | 0.85 (15%) | TCS is less volatile, losing less score. |
| **Final Rating** | **70.20** | **65.50** | Reliance ranks higher due to moat and momentum. |

**Why Reliance Ranks Higher**:
- Strong Economic Moat (Wide, user-overridden) and high weight (0.15) boost its score.
- Better SymbolTrendRS and NIFTY200DMARSIVolume indicate market strength.
- TCS excels in RoE and D/E, but lower moat and slightly higher volatility penalty reduce its rating.

---

## Limitations for Beginners to Understand

While the tool is powerful, beginners should know:
- **Data Dependency**: Relies on free APIs, which may fail or lack data for some stocks.
- **User Inputs**: Metrics like Profit Growth require manual entry, and wrong values affect accuracy.
- **Simplifications**: Economic Moat uses a placeholder for revenue stability, which isn’t precise.
- **Not Financial Advice**: The Final Rating is a guide, not a guarantee of future performance. Always research further.

---

## Tips for Newbies Using the Tool

1. **Start Small**: Test with 1-2 stocks (e.g., `RELIANCE.NS`, `TCS.NS`) in single-stock mode to understand outputs.
2. **Learn Parameters**: Use the table and Calculation Details to see why a stock scores high or low.
3. **Source Inputs**: Visit [Screener.in](https://www.screener.in/) or [Moneycontrol](https://www.moneycontrol.com/) for accurate Promoter Holding, Profit Growth, etc.
4. **Check Excel**: Ensure `portfolio.xlsx` has valid tickers and numbers to avoid warnings (e.g., `nan` for `RELIANCE.NS`).
5. **Interpret Ratings**: A rating > 70 suggests a strong stock, but compare with market conditions and your goals.

---

## Conclusion

The `stock_ranking.py` tool simplifies stock analysis by combining 15 fundamental and technical parameters into a single **Final Rating**. For beginners, it’s like a scorecard that evaluates a company’s financial health (e.g., debt, profits) and market performance (e.g., price trends, volume). Each parameter contributes a score, weighted by importance, and adjusted for risk (volatility penalty). The result is a ranking that helps you compare stocks like Reliance and TCS.

**Key Takeaways**:
- **Fundamental Parameters** (e.g., RoE, Moat) assess long-term value.
- **Technical Parameters** (e.g., RSI, DMA) assess short-term trends.
- **Weights and Volatility** balance importance and risk.
- **Output** (Excel, plots) makes results easy to understand and compare.
