### Parameter Analysis
#### Manual Input Parameters (11)
These were manually entered in the Pine Script. We’ll assess whether each can be programmatically fetched using free APIs (`yfinance`, `nsepy`, or web scraping) or requires user input:
1. **Promoter Holding (%)**: Not available via `yfinance` or `nsepy`. Can be scraped from Screener.in or Moneycontrol, but scraping is unreliable due to website changes or access restrictions. **User input recommended**.
2. **Institutional Holding (%)**: Similar to Promoter Holding, available via Screener.in (FII + DII). Scraping is possible but fragile. **User input recommended**.
3. **Debt-to-Equity Ratio**: Available via `yfinance` (`info['debtToEquity']`). Can be fetched programmatically.
4. **Return on Equity (%)**: Available via `yfinance` (`info['returnOnEquity']`). Can be fetched programmatically.
5. **Profit Growth YoY (%)**: Sometimes available via Screener.in or financial reports but not consistently via free APIs. Scraping is possible but unreliable. **User input recommended**.
6. **Profit CAGR 5Y (%)**: Not directly available via `yfinance`. Can be scraped from Screener.in or calculated from historical financials, but requires complex data fetching. **User input recommended**.
7. **Dividend Yield (%)**: Available via `yfinance` (`info['dividendYield']`). Can be fetched programmatically.
8. **Economic Moat**: Qualitative metric requiring analyst judgment (e.g., Morningstar’s moat rating). No free API provides this. **User input required**.
9. **Valuation Score (0-100)**: Subjective score based on valuation metrics (e.g., P/E, P/B). While P/E can be fetched via `yfinance`, a composite score requires user judgment or complex modeling. **User input required**.
10. **SymbolTrendRS (0-100)**: Custom technical score based on relative strength trends. Can be approximated using technical indicators, but the Pine Script used a manual input for flexibility. **User input recommended** for consistency.
11. **NIFTY200DMARSIVolume (0-100)**: Custom composite score combining NIFTY 200 performance, DMA, RSI, and volume. No standard API provides this, and it’s highly subjective. **User input required**.

#### Calculated Metrics (4)
These are computed automatically in the Pine Script and can be replicated in Python:
1. **200-Day DMA**: Calculated using historical price data (`yfinance` + `pandas`).
2. **14-Day RSI**: Calculated using `ta` library.
3. **20-Day Volume MA**: Calculated using `yfinance` volume data.
4. **Simple RS vs NIFTY_50**: Calculated by comparing stock and NIFTY 50 price changes (`yfinance`).

#### Parameters Requiring User Input
Based on the analysis, the following 7 parameters cannot be reliably calculated programmatically using free APIs and will require user input:
- **Promoter Holding (%)**
- **Institutional Holding (%)**
- **Profit Growth YoY (%)**
- **Profit CAGR 5Y (%)**
- **Economic Moat**
- **Valuation Score (0-100)**
- **NIFTY200DMARSIVolume (0-100)**

**Note**: `SymbolTrendRS` could be approximated programmatically, but since the Pine Script used manual input, we’ll keep it as user input for consistency. If you prefer, I can calculate it based on technical indicators (let me know).

### Python Code
The following Python code:
1. Prompts the user for an NSE stock ticker.
2. Fetches available fundamental data (`Debt-to-Equity Ratio`, `RoE`, `Dividend Yield`) using `yfinance`.
3. Prompts the user for the 7 parameters that cannot be calculated, providing detailed instructions on how to obtain each value.
4. Calculates technical metrics (DMA, RSI, Volume MA, RS) using `yfinance` and `ta`.
5. Computes the final rating using the same scoring system as the Pine Script.
6. Displays the results in a table using `tabulate`.
