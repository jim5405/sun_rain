# optimizer.py (Version 4, Simplified for Execution)

import pandas as pd
import yfinance as yf
import numpy as np
import random
import time
import argparse
from scipy.stats.mstats import gmean
import warnings
import sys
import os
from datetime import datetime, timedelta
import pickle

# --- 環境設定 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.scan_module import (
    calculate_indicators,
    get_barometer_status,
    get_recovery_status
)

# --- 常數設定 ---
N_TRIALS = 100
TICKERS_TO_TEST = ["0050.TW", "006208.TW", "VOO", "QQQ"]
RISK_FREE_RATE = 0.02
TRADING_DAYS_PER_YEAR = 252
CACHE_DIR = "data_cache"

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. 參數搜索空間 ---
PARAM_SPACE_FINETUNE = {
    "ma_short": list(range(30, 81, 10)), "ma_long": list(range(120, 201, 20)),
    "rsi_window": [14, 20, 25], "rsi_oversold": [30, 35],
    "rsi_bull_threshold": [50, 55], "rsi_bear_threshold": [40, 45],
    "macd_fast": [12, 15], "macd_slow": [26, 30], "macd_signal": [9, 12, 18],
    "drawdown_window": list(range(200, 301, 50)), "drawdown_no_rain": [-0.10, -0.12],
    "adx_period": [14, 20], "adx_threshold": [18, 20, 22],
    "bb_window": [20, 25], "bb_std_dev": [2, 2.5]
}

# --- 核心回測功能函數 ---

def get_stock_data_from_cache(ticker, years=5):
    cache_file = os.path.join(CACHE_DIR, f"{ticker}_{years}y.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        # 如果快取不存在，直接返回 None，因為此版本不應再下載
        print(f"錯誤: 找不到 {ticker} 的快取數據。請先執行可下載數據的版本。")
        return None

def calculate_sharpe_ratio(returns):
    excess_returns = returns - RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    if np.std(excess_returns) == 0: return 0.0
    return np.sqrt(TRADING_DAYS_PER_YEAR) * (excess_returns.mean() / np.std(excess_returns))

def run_backtest_for_optimizer(df, config):
    df_calc = calculate_indicators(df.copy(), config)
    df_calc['Barometer_Status'] = df_calc.apply(get_barometer_status, axis=1, config=config)
    recovery_statuses = [get_recovery_status(df_calc.iloc[i], df_calc.iloc[i-1] if i > 0 else None, config) for i in range(len(df_calc))]
    df_calc['Recovery_Status'] = recovery_statuses
    df_calc = df_calc.dropna()
    if len(df_calc) < 2: return None

    capital = 1.0; position, trades, win_trades, buy_price = 0, 0, 0, 0
    daily_returns = []

    for i in range(1, len(df_calc)):
        row, prev_row = df_calc.iloc[i], df_calc.iloc[i-1]
        if position == 1: daily_returns.append((row['Close'] / prev_row['Close']) - 1)
        else: daily_returns.append(0)

        if (row['Recovery_Status'] == '撥雲見日') and position == 0:
            position = 1; buy_price = row['Close']
        elif ("雨天" in row['Barometer_Status'] or "颱風天" in row['Barometer_Status']) and position == 1:
            position = 0; trades += 1; profit = (row['Close'] - buy_price) / buy_price
            if profit > 0: win_trades += 1
            capital *= (1 + profit)

    if position == 1:
        trades += 1; profit = (df_calc.iloc[-1]['Close'] - buy_price) / buy_price
        if profit > 0: win_trades += 1; capital *= (1 + profit)

    total_return = capital - 1.0
    daily_returns_series = pd.Series(daily_returns, index=df_calc.index[1:])
    num_years = len(df_calc) / TRADING_DAYS_PER_YEAR
    annualized_return = (1 + total_return) ** (1/num_years) - 1 if num_years > 0 and total_return > -1 else 0

    return { "annualized_return": annualized_return, "sharpe_ratio": calculate_sharpe_ratio(daily_returns_series), "win_rate": (win_trades / trades) if trades > 0 else 0 }

def evaluate_config(config, tickers_data):
    results = [run_backtest_for_optimizer(df, config) for df in tickers_data.values() if df is not None]
    results = [r for r in results if r is not None]
    if not results: return None
    
    annualized_returns = [1 + r['annualized_return'] for r in results if r['annualized_return'] > -1]
    sharpe_ratios = [r['sharpe_ratio'] for r in results]
    win_rates = [r['win_rate'] for r in results]
    
    if not annualized_returns: return None

    return { "avg_annualized_return": gmean(annualized_returns) - 1, "avg_sharpe_ratio": np.mean(sharpe_ratios), "avg_win_rate": np.mean(win_rates) }

# --- 3. 主優化循環 ---
def main():
    print("===== 正在从快取中读取数据... =====")
    all_data = {ticker: get_stock_data_from_cache(ticker) for ticker in TICKERS_TO_TEST}

    if any(df is None for df in all_data.values()):
        print("错误：部分快取数据缺失，无法继续。")
        return

    print("\n===== 开始二次参数优化 (目标: max_sharpe) =====")
    best_score, best_config, best_details = -float('inf'), None, {}
    
    for i in range(N_TRIALS):
        trial_config = {key: random.choice(values) for key, values in PARAM_SPACE_FINETUNE.items()}
        if trial_config['ma_short'] >= trial_config['ma_long']: continue
            
        print(f"\r--- 试验 [{i+1}/{N_TRIALS}] ---", end="")
        details = evaluate_config(trial_config, all_data)

        if details:
            score = details['avg_sharpe_ratio']
            if score > best_score:
                best_score, best_config, best_details = score, trial_config, details
                print(f"\n🎉 新最佳解! 年化報酬: {details['avg_annualized_return']:.2%}, 夏普率: {details['avg_sharpe_ratio']:.2f}, 勝率: {details['avg_win_rate']:.2%}")

    print("\n\n" + "="*40)
    print(f"      🎉 二次优化完成！(目标: max_sharpe)")
    print("="*40)
    if best_config:
        print(f"综合平均年化报酬率: {best_details.get('avg_annualized_return', 0):.2%}")
        print(f"综合平均夏普比率: {best_details.get('avg_sharpe_ratio', 0):.2f}")
        print(f"综合平均胜率: {best_details.get('avg_win_rate', 0):.2%}")
        print("\n最佳参数设定:")
        print("CONFIG = {")
        for key, value in sorted(best_config.items()):
             print(f"    \"{key}\": {value},")
        print("}")
    else:
        print("未能在本次优化中找到有效的参数组合。")

if __name__ == '__main__':
    main()
