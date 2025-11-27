# Project Structure and Architecture Analysis

## 1. Project Overview
This project is a stock analysis and scanning system designed to identify trading opportunities based on technical indicators. It supports multi-threading for efficient scanning of large lists of stocks (US and Taiwan markets), provides backtesting capabilities, and includes tools for parameter optimization.

## 2. File Structure
```
d:\sun_rain
├── Auto_scan.py                # Main entry point for automated stock scanning
├── scan_module.py              # Core module containing analysis logic and data fetching
├── stock_analysis.py           # Single stock analysis with visualization (Plotly)
├── judge_stock.py              # CLI tool for quick diagnosis of a single stock
├── optimizer.py                # Strategy parameter optimization tool
├── batch_tester.py             # Batch backtesting tool for strategy validation
├── Add_delete_hold.py          # Utility to manage the hold list
├── test_suite.py               # Test suite for the project
├── Model_conf.py               # Default model configuration
├── Model_conf_aggressive.py    # Aggressive strategy configuration
├── Model_conf_alt.py           # Alternative model configuration
└── hold_list.txt               # Text file storing user's held tickers
```

## 3. Module Descriptions

### Core Components

#### `scan_module.py`
The backbone of the system. It encapsulates:
- **Data Acquisition**: Fetches stock data using `yfinance`.
- **Indicator Calculation**: Computes MA, RSI, MACD, Drawdown, ADX, etc.
- **Status Determination**:
    - `get_barometer_status`: Determines market state (Sunny, Cloudy, Rainy, etc.) based on MA and RSI.
    - `get_recovery_status`: Identifies recovery signals ("撥雲見日") based on Drawdown, MACD, and ADX.
- **Stock Lists**: Manages lists of tickers (SP500, TWSE).

#### `Auto_scan.py`
The primary execution script for daily operations.
- **Multi-threading**: Uses `ThreadPoolExecutor` to scan multiple stocks concurrently.
- **Dual Model Support**: Can run two models (e.g., Default and Alt) simultaneously for cross-verification.
- **Reporting**: Outputs a console report classifying stocks into "Held", "Buy Opportunities", and "Sell/Reduce Opportunities".

### Analysis & Research Tools

#### `stock_analysis.py`
Focused on deep-dive analysis of a single stock.
- **Visualization**: Generates interactive HTML charts using Plotly, showing Price, MA, Buy/Sell points, RSI, MACD, and Drawdown.
- **Backtesting**: Runs "Timing" and "DCA (Dollar Cost Averaging)" backtests to validate strategy performance.

#### `optimizer.py`
Used to find the best parameters for the strategy.
- **Genetic/Random Search**: Iterates through parameter spaces (MA lengths, RSI thresholds, etc.) to maximize Return or Win Rate.
- **Configuration**: Supports optimizing for "Conservative" or "Aggressive" profiles.

#### `batch_tester.py`
Runs the strategy against a predefined list of major tickers to gauge overall performance.
- **Comparison**: Compares Strategy Return vs. Buy & Hold Return.

#### `judge_stock.py`
A lightweight CLI tool to quickly check the status (Barometer & Recovery) of a specific ticker without running a full scan.

## 4. Data Flow

1.  **Configuration**: Parameters are loaded from `Model_conf*.py` files.
2.  **Input**:
    - `Auto_scan.py` gets tickers from `scan_module.get_dynamic_scan_list()` and `hold_list.txt`.
    - `judge_stock.py` takes a ticker argument.
3.  **Processing** (`scan_module.py`):
    - `get_stock_data()`: Fetches OHLCV data.
    - `calculate_indicators()`: Adds technical indicators to the DataFrame.
    - `get_barometer_status()` & `get_recovery_status()`: Evaluates the latest market condition.
4.  **Output**:
    - `Auto_scan.py`: Prints a summary report to the console.
    - `stock_analysis.py`: Generates an HTML report file.

## 5. Key Logic

- **Barometer (Market State)**:
    - **Sunny (☀️)**: Price > Short MA > Long MA & RSI > Bull Threshold.
    - **Rainy (🌧️)**: Short MA > Price & Long MA > Price & RSI < Bear Threshold.
- **Recovery (Signal)**:
    - **Clearing Up (撥雲見日)**: Drawdown improves, MACD Histogram turns positive, and ADX confirms trend strength.
