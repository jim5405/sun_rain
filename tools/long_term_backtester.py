# long_term_backtester.py

import pandas as pd
import yfinance as yf
import numpy as np
import time
import argparse
import importlib
import sys
import os
from datetime import datetime, timedelta

# --- 環境設定 ---
# 將專案根目錄加入 sys.path 以便導入核心模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 從核心模組導入最新的函式
from core.scan_module import (
    calculate_indicators,
    get_barometer_status,
    get_recovery_status
)

# --- 常數設定 ---
# 預設回測的股票列表
DEFAULT_TICKERS = ["2330.TW", "AAPL", "0050.TW", "QQQ"]
# 年化無風險利率，用於計算夏普比率
RISK_FREE_RATE = 0.02
# 每年交易天數的估計值
TRADING_DAYS_PER_YEAR = 252

# --- 核心功能函數 ---

def get_stock_data_for_backtest(ticker, years=10):
    """下載指定年份的股票歷史數據"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365.25)

    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date, auto_adjust=True)

    if df.empty:
        print(f"警告: 無法下載 {ticker} 在指定期間的數據。")
        return None
    time.sleep(0.2)
    return df

def calculate_sharpe_ratio(returns):
    """計算夏普比率"""
    excess_returns = returns - RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    # 如果標準差為零（例如沒有交易），則夏普比率為零
    if np.std(excess_returns) == 0:
        return 0.0
    return np.sqrt(TRADING_DAYS_PER_YEAR) * (excess_returns.mean() / np.std(excess_returns))

def calculate_max_drawdown(cumulative_returns):
    """計算最大回撤"""
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def run_long_term_backtest(df, config):
    """執行單一標的的長期回測並計算詳細績效指標"""
    df_calc = calculate_indicators(df.copy(), config)
    df_calc['Barometer_Status'] = df_calc.apply(get_barometer_status, axis=1, config=config)

    recovery_statuses = []
    for i in range(len(df_calc)):
        status = get_recovery_status(df_calc.iloc[i], df_calc.iloc[i-1] if i > 0 else None, config)
        recovery_statuses.append(status)
    df_calc['Recovery_Status'] = recovery_statuses

    df_calc = df_calc.dropna()
    if len(df_calc) < 2: return None

    # --- 交易模擬 ---
    capital = 1.0
    position = 0
    trades = 0
    win_trades = 0
    buy_price = 0
    daily_returns = []

    for i in range(1, len(df_calc)):
        row = df_calc.iloc[i]
        prev_row = df_calc.iloc[i-1]

        buy_condition = row['Recovery_Status'] == '撥雲見日'
        sell_condition = "雨天" in row['Barometer_Status'] or "颱風天" in row['Barometer_Status']

        # 計算每日報酬率
        if position == 1:
            daily_returns.append((row['Close'] / prev_row['Close']) - 1)
        else:
            daily_returns.append(0)

        # 執行交易
        if buy_condition and position == 0:
            position = 1
            buy_price = row['Close']
        elif sell_condition and position == 1:
            position = 0
            trades += 1
            profit = (row['Close'] - buy_price) / buy_price
            if profit > 0: win_trades += 1
            capital *= (1 + profit)

    # 如果回測結束時仍持有部位，則以最後一天的收盤價賣出
    if position == 1:
        trades += 1
        profit = (df_calc['Close'].iloc[-1] - buy_price) / buy_price
        if profit > 0: win_trades += 1
        capital *= (1 + profit)

    # --- 績效計算 ---
    total_return = capital - 1.0
    buy_and_hold_return = (df_calc['Close'].iloc[-1] / df_calc['Close'].iloc[0]) - 1

    daily_returns_series = pd.Series(daily_returns, index=df_calc.index[1:])
    cumulative_returns = (1 + daily_returns_series).cumprod()

    num_years = len(df_calc) / TRADING_DAYS_PER_YEAR
    annualized_return = (1 + total_return) ** (1/num_years) - 1 if num_years > 0 else 0

    return {
        "b&h_return": buy_and_hold_return,
        "strategy_total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": calculate_sharpe_ratio(daily_returns_series),
        "max_drawdown": calculate_max_drawdown(cumulative_returns),
        "win_rate": (win_trades / trades) if trades > 0 else 0,
        "trades": trades,
    }

def load_config(model_name):
    """載入指定的模型設定檔"""
    try:
        module = importlib.import_module(f"config.{model_name}")
        return module.CONFIG
    except ImportError:
        print(f"錯誤: 找不到模型設定檔 'config/{model_name}.py'")
        exit(1)

# --- 主程式 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="長期策略回測與分析工具")
    parser.add_argument("--model", type=str, default="Model_conf", help="要使用的模型設定檔。")
    parser.add_argument("--tickers", nargs='+', default=DEFAULT_TICKERS, help="要進行回測的股票列表。")
    parser.add_argument("--years", type=int, default=10, help="回測的年份長度。")
    args = parser.parse_args()

    CONFIG = load_config(args.model)

    all_results = []
    print(f"===== 開始進行 {args.years} 年長期回測 (模型: {args.model}) =====")
    print(f"測試標的: {', '.join(args.tickers)}")

    for ticker in args.tickers:
        print(f"\n--- 正在處理: {ticker} ---")
        try:
            df = get_stock_data_for_backtest(ticker, args.years)
            if df is None: continue

            result = run_long_term_backtest(df, CONFIG)

            if result:
                result['ticker'] = ticker
                all_results.append(result)
                print(f"  > 策略總報酬率: {result['strategy_total_return']:.2%}")
                print(f"  > B&H 總報酬率: {result['b&h_return']:.2%}")
                print(f"  > 年化報酬率: {result['annualized_return']:.2%}")
                print(f"  > 夏普比率: {result['sharpe_ratio']:.2f}")
                print(f"  > 最大回撤: {result['max_drawdown']:.2%}")
                print(f"  > 交易次數: {result['trades']}")
                print(f"  > 策略勝率: {result['win_rate']:.2%}")
            else:
                print(f"  > 未能產生有效的回測結果。")
        except Exception as e:
            print(f"處理 {ticker} 時發生嚴重錯誤: {e}")

    # --- 產生總結報告 ---
    if all_results:
        summary_df = pd.DataFrame(all_results)
        print("\n\n" + "="*80)
        print(f"                              長期回測績效總結 ({args.model})")
        print("="*80)
        print(summary_df.to_string(
            columns=['ticker', 'strategy_total_return', 'b&h_return', 'annualized_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'trades'],
            formatters={
                'strategy_total_return': '{:.2%}'.format,
                'b&h_return': '{:.2%}'.format,
                'annualized_return': '{:.2%}'.format,
                'sharpe_ratio': '{:.2f}'.format,
                'max_drawdown': '{:.2%}'.format,
                'win_rate': '{:.2%}'.format
            }
        ))
        print("\n" + "="*80)

        # 計算綜合平均指標
        avg_metrics = summary_df.drop(columns='ticker').mean()
        print("\n[綜合平均績效]")
        print(f"平均年化報酬率: {avg_metrics['annualized_return']:.2%}")
        print(f"平均夏普比率: {avg_metrics['sharpe_ratio']:.2f}")
        print(f"平均最大回撤: {avg_metrics['max_drawdown']:.2%}")
        print(f"平均勝率: {avg_metrics['win_rate']:.2%}")
        print(f"平均年交易次數: {avg_metrics['trades'] / args.years:.1f}")
    else:
        print("\n無任何有效回測結果可供總結。")
