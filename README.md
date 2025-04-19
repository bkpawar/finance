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

