
### Objective
The code aims to rank stocks based on a combination of **fundamental analysis** (assessing a company’s financial health) and **technical analysis** (evaluating price trends and market signals). The Final Rating (0-100) reflects a stock’s investment potential, with higher scores indicating stronger fundamentals, positive momentum, and lower risk.

### Parameters and Rationale
The 15 parameters are chosen to capture a company’s financial strength, growth potential, valuation, and market momentum. Each is scored (0-100) and weighted to compute the Final Rating.

1. **Promoter Holding (%)**:
   - **Theory**: High promoter ownership signals confidence in the company’s future and aligns management with shareholders.
   - **Calculation**: User input; scaled as `min(100, value * 2)`.
   - **Weight**: 10% (significant but not dominant).

2. **Institutional Holding (%)**:
   - **Theory**: Strong FII/DII ownership indicates professional investor confidence.
   - **Calculation**: User input; scaled as `min(100, value * 4)`.
   - **Weight**: 10%.

3. **Debt-to-Equity Ratio**:
   - **Theory**: Lower leverage reduces financial risk.
   - **Calculation**: Fetched from `yahooquery` or `yfinance`; scored as `max(0, 100 - value * 50)`.
   - **Weight**: 10%.

4. **Return on Equity (RoE)**:
   - **Theory**: Measures profitability relative to equity; higher RoE indicates efficient capital use.
   - **Calculation**: Fetched; scored as `min(100, value * 5)`.
   - **Weight**: 10%.

5. **Profit Growth YoY (%)**:
   - **Theory**: Strong year-over-year profit growth signals business momentum.
   - **Calculation**: User input; scored as `min(100, value * 5)`.
   - **Weight**: 10%.

6. **Profit CAGR 5Y (%)**:
   - **Theory**: Consistent long-term profit growth indicates stability.
   - **Calculation**: User input; scored as `min(100, value * 5)`.
   - **Weight**: 10%.

7. **Dividend Yield (%)**:
   - **Theory**: Higher yields provide income and signal financial health.
   - **Calculation**: Fetched; scored as `min(100, value * 50)`.
   - **Weight**: 5% (less critical for growth stocks).

8. **Economic Moat**:
   - **Theory**: A competitive advantage (e.g., brand, patents) protects long-term profitability.
   - **Calculation**: Based on profit margin (>20%), market cap (>₹1T), ROIC (>15%), revenue stability (<5%). Scored as Wide=100, Narrow=50, None=0.
   - **Weight**: 5% (qualitative but impactful).

9. **Valuation Score**:
   - **Theory**: Lower P/E and P/B ratios suggest undervaluation.
   - **Calculation**: Average of P/E score (`100 - (P/E / industry P/E) * 50`) and P/B score (`100 - (P/B / 5) * 50`).
   - **Weight**: 15% (key for value investing).

10. **SymbolTrendRS**:
    - **Theory**: Combines RSI and MACD to assess short-term momentum.
    - **Calculation**: Average of RSI score (`(RSI / 100) * 50`) and MACD score (50 if positive, 25 if not).
    - **Weight**: 10%.

11. **NIFTY200DMARSIVolume**:
    - **Theory**: Composite metric for technical strength (trend, momentum, liquidity, outperformance).
    - **Calculation**: Sum of DMA (25 if price > DMA), RSI (25 if 30≤RSI≤70), volume (25 if above average), RS (25 if >50).
    - **Weight**: 10%.

12. **200-Day DMA**:
    - **Theory**: Price above 200-day DMA indicates a bullish trend.
    - **Calculation**: Scored as 100 if price > DMA, else 0.
    - **Weight**: 5%.

13. **14-Day RSI**:
    - **Theory**: RSI measures momentum; neutral (30-70) is stable, overbought (>70) or oversold (<30) signal reversals.
    - **Calculation**: Scored as 75 if >70, 25 if <30, else 50.
    - **Weight**: 5%.

14. **20-Day Volume MA**:
    - **Theory**: Higher volume suggests strong investor interest.
    - **Calculation**: Scored as 75 if current volume > average, else 25.
    - **Weight**: 5%.

15. **Simple RS vs NIFTY_50**:
    - **Theory**: Outperformance vs. NIFTY 50 indicates relative strength.
    - **Calculation**: Stock’s 63-day return minus NIFTY 50’s, scaled to 0-100.
    - **Weight**: 10%.

### Volatility Penalty
- **Theory**: High volatility increases risk, reducing risk-adjusted returns.
- **Calculation**: `max(0, 100 - (volatility * 100)) / 100`, where volatility is the annualized standard deviation of daily returns.
- **Impact**: Multiplies each score to penalize volatile stocks.

### Scoring and Weighting
- Each parameter is scaled to 0-100 for consistency.
- Weights (from `config.ini`) prioritize valuation (15%), core fundamentals (10%), and technicals (5-10%).
- Final Rating = Weighted sum of scores / Sum of weights, adjusted by volatility penalty.

### Why This Approach?
- **Fundamental Analysis**: Ensures financial health (e.g., low debt, high RoE) and growth (e.g., profit CAGR).
- **Technical Analysis**: Captures market sentiment and momentum (e.g., RSI, DMA).
- **Risk Adjustment**: Volatility penalty favors stable stocks.
- **Flexibility**: User inputs and moat override allow customization.
- **Automation**: Minimizes manual effort by calculating 11 parameters.

---

## 4. Further Reading Links

### Financial Concepts
- **Fundamental Analysis**:
  - [Investopedia: Fundamental Analysis](https://www.investopedia.com/terms/f/fundamentalanalysis.asp)
  - [Morningstar: Understanding Economic Moat](https://www.morningstar.com/articles/1046146/what-is-an-economic-moat)
- **Technical Analysis**:
  - [Investopedia: Technical Analysis](https://www.investopedia.com/terms/t/technicalanalysis.asp)
  - [StockCharts: Introduction to Technical Indicators](https://school.stockcharts.com/doku.php?id=technical_indicators)
- **Key Metrics**:
  - [Investopedia: Return on Equity (ROE)](https://www.investopedia.com/terms/r/returnonequity.asp)
  - [Investopedia: Debt-to-Equity Ratio](https://www.investopedia.com/terms/d/debtequityratio.asp)
  - [Investopedia: Price-to-Earnings (P/E) Ratio](https://www.investopedia.com/terms/p/price-earningsratio.asp)
  - [Investopedia: Relative Strength Index (RSI)](https://www.investopedia.com/terms/r/rsi.asp)
  - [Investopedia: Moving Average Convergence Divergence (MACD)](https://www.investopedia.com/terms/m/macd.asp)

### Python Libraries
- **yahooquery**: [Official Documentation](https://yahooquery.dpguthrie.com/)
- **yfinance**: [GitHub Repository](https://github.com/ranaroussi/yfinance)
- **ta (Technical Analysis)**: [GitHub Repository](https://github.com/bukosabino/ta)
- **pandas**: [Official Documentation](https://pandas.pydata.org/docs/)
- **numpy**: [Official Documentation](https://numpy.org/doc/stable/)
- **matplotlib**: [Official Documentation](https://matplotlib.org/stable/contents.html)
- **prompt_toolkit**: [Official Documentation](https://python-prompt-toolkit.readthedocs.io/en/stable/)
- **tabulate**: [GitHub Repository](https://github.com/astanin/python-tabulate)

### Stock Market and NSE
- [NSE India](https://www.nseindia.com/): Official site for NSE data and tickers.
- [Screener.in](https://www.screener.in/): Source for promoter holding, profit growth, etc.
- [Moneycontrol](https://www.moneycontrol.com/): Financial data and shareholding patterns.

### Investment Strategies
- [The Intelligent Investor by Benjamin Graham](https://www.amazon.com/Intelligent-Investor-Definitive-Investing-Essentials/dp/0060555661): Classic book on value investing.
- [A Random Walk Down Wall Street by Burton Malkiel](https://www.amazon.com/Random-Walk-Down-Wall-Street/dp/0393358380): Overview of investment strategies.
- [Morningstar: Stock Investing](https://www.morningstar.com/stocks): Articles on stock analysis.

---

## 5. Additional Notes

### Limitations
- **Data Gaps**: Free APIs (`yahooquery`, `yfinance`) may miss data; defaults are used.
- **Revenue Stability**: Placeholder (10%) due to lack of historical revenue data.
- **Industry P/E**: Static (30); dynamic data would improve Valuation Score.
- **User Inputs**: Promoter and Institutional Holding require manual entry; paid APIs could automate these.

### Future Improvements
- **Paid APIs**: Use Financial Modeling Prep or Alpha Vantage for complete data.
- **Backtesting**: Validate the Final Rating’s predictive power with historical returns.
- **Sentiment Analysis**: Incorporate news sentiment via `newsapi`.
- **Portfolio Analysis**: Analyze multiple stocks for diversification.

### Recommendations
- **Run Regularly**: Analyze multiple stocks to build a diversified portfolio.
- **Verify Inputs**: Use Screener.in or Moneycontrol for accurate user inputs.
- **Adjust Weights**: Modify `config.ini` based on investment priorities (e.g., higher weight for Valuation Score).

If you need further assistance (e.g., debugging, adding features, or backtesting), please provide details, and I’ll tailor the response!



