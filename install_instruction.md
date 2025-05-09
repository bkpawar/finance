### Installation Instructions
Install the required Python libraries:
```bash
pip install yahooquery yfinance pandas numpy ta tabulate prompt_toolkit matplotlib openpyxl
or
pip3 install yahooquery yfinance pandas numpy ta tabulate prompt_toolkit matplotlib openpyxl
```
### Code Structure
- **Imports**: Libraries for data fetching (`yahooquery`, `yfinance`), technical analysis (`ta`), data handling (`pandas`, `numpy`), user interface (`prompt_toolkit`, `tabulate`), plotting (`matplotlib`), and configuration (`configparser`).
- **StockAnalyzer Class**:
  - `__init__`: Initializes ticker, configuration, and data storage.
  - `fetch_fundamental_data`: Retrieves fundamental metrics and calculates Economic Moat and Valuation Score.
  - `prompt_user_for_inputs`: Collects 4 user inputs and allows moat override.
  - `calculate_technical_metrics`: Computes technical indicators and custom metrics.
  - `calculate_final_rating`: Calculates weighted scores with volatility penalty.
  - `display_table`: Outputs results and calculation details.
  - `plot_metrics`: Generates price vs. DMA plot.
- **Main Function**: Orchestrates the analysis process with user input and error handling.

---

## 2. Steps to Run

### Prerequisites
- **Python Version**: Python 3.8 or higher.
- **Operating System**: macOS, Windows, or Linux.
- **Internet Connection**: Required for API data fetching.

### Step-by-Step Instructions
1. **Install Dependencies**:
   - Install required Python libraries using pip:
     ```bash
     pip3 install yahooquery yfinance pandas numpy ta tabulate prompt_toolkit matplotlib openpyxl
     ```

2. **Save the Code**:
   - Copy the code above into a file named `stock_ranking.py`.
   - Save it in a directory (e.g., `/Users/your_username/finance/`).

3. **Create Configuration File**:
   - Create a file named `config.ini` in the same directory as `stock_ranking.py`.
   - Add the following content:
     ```ini
     [DEFAULT]
     benchmark_ticker = ^NSEI
     dma_length = 200
     rsi_length = 14
     vol_avg_length = 20
     rs_length = 63
     weights = 0.1,0.1,0.1,0.1,0.1,0.1,0.05,0.05,0.15,0.1,0.1,0.05,0.05,0.05,0.1
     ```
   - This configures the benchmark index (NIFTY 50), technical indicator periods, and scoring weights.

4. **Run the Script**:
   - Open a terminal and navigate to the directory:
     ```bash
     cd /Users/your_username/finance
     ```
   - Execute the script:
     ```bash
     python stock_ranking.py
     ```

5. **Provide Inputs**:
   - **Ticker**: Enter an NSE stock ticker (e.g., `RELIANCE.NS`). The script appends `.NS` if omitted.
   - **User Inputs**:
     - **Promoter Holding (%)**: Percentage of shares owned by promoters (e.g., from Screener.in).
     - **Institutional Holding (%)**: Percentage owned by FII + DII.
     - **Profit Growth YoY (%)**: Year-over-year profit growth.
     - **Profit CAGR 5Y (%)**: 5-year profit CAGR.
     - Press Enter to use defaults if unsure.
   - **Economic Moat Override**: Review the calculated moat (e.g., `Narrow`) and its basis, then accept or override with `None`, `Narrow`, or `Wide`.

6. **Review Output**:
   - **Console**: Displays a table with all parameters, calculation details for auto-calculated metrics, and a Final Rating.
   - **Files**:
     - `{ticker}_analysis.csv`: Table data in CSV format.
     - `{ticker}_plot.png`: Plot of stock price vs. 200-Day DMA.

### Example Interaction
```
Steps to Run

### Prerequisites
- **Python**: 3.8 or higher.
- **Libraries**:
  ```bash
  pip install yahooquery yfinance pandas numpy ta tabulate prompt_toolkit matplotlib openpyxl
  ```

### Setup
1. **Save the Code**:
   - Save the updated code as `stock_ranking.py`.

2. **Create `config.ini`**:
   - In the same directory, create `config.ini`:
     ```ini
     [DEFAULT]
     benchmark_ticker = ^NSEI
     dma_length = 200
     rsi_length = 14
     vol_avg_length = 20
     rs_length = 63
     weights = 0.1,0.1,0.1,0.1,0.1,0.1,0.05,0.05,0.15,0.1,0.1,0.05,0.05,0.05,0.1
     ```

3. **Prepare Input Excel**:
   - Create `portfolio.xlsx` with at least a `Ticker` column. Example:
     ```excel
     | Ticker       | Promoter Holding (%) | Inst. Holding (%) | Profit Growth YoY (%) | Profit CAGR 5Y (%) | Economic Moat |
     |--------------|---------------------|-------------------|-----------------------|--------------------|---------------|
     | RELIANCE.NS  | 50.6                | 38.5              |                       |                    | Wide          |
     | TCS.NS       | 72.3                | 23.4              | 9.0                   | 10.2               |               |
     ```
   - Ensure `Profit Growth YoY (%)` and `Profit CAGR 5Y (%)` are provided for `RELIANCE.NS` to avoid warnings. Sources: [Screener.in](https://www.screener.in/) or [Moneycontrol](https://www.moneycontrol.com/).

### Running the Code
1. **Portfolio Mode**:
   - Run with:
     ```bash
     python stock_ranking.py --portfolio portfolio.xlsx
     ```
   - **Output**:
     - `portfolio_rankings.xlsx`: Rankings and calculation details.
     - `plots/`: PNG plots for each stock (e.g., `plots/RELIANCE.NS_plot.png`).
     - Console shows progress and warnings:
       ```
       Processing RELIANCE.NS...
       Processing TCS.NS...
       Warning: Missing inputs in Excel file (defaults used):
       - RELIANCE.NS: Missing profit_growth, cagr
       Results saved to portfolio_rankings.xlsx
       ```

2. **Single Stock Mode**:
   - Run without arguments:
     ```bash
     python stock_ranking.py
     ```
   - Follow interactive prompts.

### Example Input Excel Fix
To eliminate the warnings for `RELIANCE.NS`, update `portfolio.xlsx`:
```excel
| Ticker       | Promoter Holding (%) | Inst. Holding (%) | Profit Growth YoY (%) | Profit CAGR 5Y (%) | Economic Moat |
|--------------|---------------------|-------------------|-----------------------|--------------------|---------------|
| RELIANCE.NS  | 50.6                | 38.5              | 11.5                  | 8.7                | Wide          |
| TCS.NS       | 72.3                | 23.4              | 9.0                   | 10.2               |               |
```
- Values for `Profit Growth YoY (%)` and `Profit CAGR 5Y (%)` can be sourced from financial websites or reports.



