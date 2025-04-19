### Steps for Users to Calculate Parameters
The code provides detailed instructions for each user-input parameter, but here’s a consolidated guide for clarity:

1. **Promoter Holding (%)**:
   - **Source**: Screener.in, Moneycontrol, or NSE/BSE websites under “Shareholding Pattern.”
   - **Steps**:
     1. Visit Screener.in and search for the stock (e.g., `RELIANCE`).
     2. Find the “Shareholding Pattern” section.
     3. Note the percentage under “Promoter Holding” (e.g., 50.6%).
     4. Enter this value (e.g., `50.6`).
   - **Alternative**: Check the company’s latest annual report or quarterly filings on NSE/BSE.
   - **If Unknown**: Use the default (`50.0`).

2. **Institutional Holding (%)**:
   - **Source**: Screener.in or Moneycontrol under “Shareholding Pattern.”
   - **Steps**:
     1. On Screener.in, locate “FII Holding” and “DII Holding.”
     2. Sum the percentages (e.g., FII = 20%, DII = 15% → 35%).
     3. Enter the total (e.g., `35.0`).
   - **Alternative**: Use NSE/BSE filings or financial portals like Yahoo Finance (if available).
   - **If Unknown**: Use the default (`15.0`).

3. **Profit Growth YoY (%)**:
   - **Source**: Screener.in, Moneycontrol, or company financial reports.
   - **Steps**:
     1. Find “Net Profit” or “PAT” for the current and previous year (e.g., via Screener.in’s “Profit & Loss” section).
     2. Calculate: `((Current Year Profit - Previous Year Profit) / Previous Year Profit) * 100`.
     3. Example: Current profit = ₹120 crore, previous = ₹100 crore → `(120 - 100) / 100 * 100 = 20%`.
     4. Enter the value (e.g., `20.0`).
   - **Alternative**: Check quarterly results or analyst reports for YoY growth.
   - **If Unknown**: Use the default (`10.0`).

4. **Profit CAGR 5Y (%)**:
   - **Source**: Screener.in or financial reports.
   - **Steps**:
     1. Find net profit for the current year and 5 years ago (e.g., via Screener.in’s “Profit & Loss”).
     2. Calculate: `((Ending Profit / Starting Profit)^(1/5) - 1) * 100`.
     3. Example: Profit 5 years ago = ₹50 crore, now = ₹80 crore → `((80/50)^(1/5) - 1) * 100 ≈ 9.86%`.
     4. Enter the value (e.g., `9.86`).
   - **Alternative**: Use Screener.in’s “Profit CAGR 5 Years” if available.
   - **If Unknown**: Use the default (`12.0`).

5. **Economic Moat**:
   - **Source**: Analyst reports (e.g., Morningstar) or your own analysis.
   - **Steps**:
     1. Assess the company’s competitive advantage:
        - **Wide**: Strong, durable advantage (e.g., Apple’s brand, Reliance’s market dominance).
        - **Narrow**: Some advantage, less durable (e.g., regional banks).
        - **None**: Commodity businesses (e.g., small steel producers).
     2. Example: For Reliance Industries, select “Narrow” due to its diversified but competitive sectors.
     3. Choose “None,” “Narrow,” or “Wide.”
   - **Alternative**: Research industry reports or competitor analysis.
   - **If Unknown**: Use the default (“Narrow”).

6. **Valuation Score (0-100)**:
   - **Source**: Valuation metrics (P/E, P/B) from Screener.in, Yahoo Finance, or DCF models.
   - **Steps**:
     1. Compare the stock’s P/E or P/B ratio to its industry average (e.g., via Screener.in).
     2. Assign a score:
        - High (70-100): Undervalued (low P/E vs. peers).
        - Medium (40-70): Fairly valued.
        - Low (0-40): Overvalued.
     3. Example: If P/E is below industry average, enter `80.0`.
     4. Enter the value (e.g., `80.0`).
   - **Alternative**: Use analyst target prices or DCF calculators online.
   - **If Unknown**: Use the default (`50.0`).

7. **NIFTY200DMARSIVolume (0-100)**:
   - **Source**: Technical analysis tools (e.g., TradingView, StockCharts).
   - **Steps**:
     1. Analyze:
        - **NIFTY 200 Comparison**: Is the stock outperforming the NIFTY 200 index?
        - **DMA**: Is the price above the 200-day DMA?
        - **RSI**: Is RSI in the 30-70 range?
        - **Volume**: Is volume above average?
     2. Assign a score (e.g., 25 points per positive factor):
        - Example: Outperforms NIFTY 200, above DMA, RSI = 60, high volume → `80.0`.
     3. Enter the value (e.g., `80.0`).
   - **Alternative**: Use TradingView’s technical indicators or manual chart analysis.
   - **If Unknown**: Use the default (`50.0`).
