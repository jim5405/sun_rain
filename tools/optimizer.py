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

# 將專案根目錄加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- 從核心模組導入最新的函式 ---
from core.scan_module import (
    get_stock_data,
    calculate_indicators,
    get_barometer_status,
    get_recovery_status
)

# --- 優化器設定 ---
N_TRIALS = 100
TICKERS_TO_TEST = ["0050.TW", "006208.TW", "VOO", "QQQ"]

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. 參數搜索空間 (Search Space) ---
# 加入布林通道參數
PARAM_SPACE_CONSERVATIVE = {
    "ma_short": list(range(40, 91, 10)),
    "ma_long": list(range(150, 251, 30)),
    "rsi_window": [20, 25, 30],
    "rsi_oversold": [30, 35, 40],
    "rsi_bull_threshold": [50, 55, 60],
    "rsi_bear_threshold": [40, 45, 50],
    "macd_fast": [12, 15, 18],
    "macd_slow": [26, 30, 35],
    "macd_signal": [9, 12, 18], 
    "drawdown_window": list(range(250, 351, 50)),
    "drawdown_no_rain": [-0.10, -0.12, -0.15],
    "adx_period": [14, 20, 25],
    "adx_threshold": [20, 25, 30],
    "bb_window": [20, 25, 30],
    "bb_std_dev": [2, 2.5]
}

PARAM_SPACE_AGGRESSIVE = {
    "ma_short": list(range(20, 61, 10)),
    "ma_long": list(range(80, 181, 20)),
    "rsi_window": [7, 10, 14, 20],
    "rsi_oversold": [20, 25, 30],
    "rsi_bull_threshold": [55, 60, 65],
    "rsi_bear_threshold": [45, 50, 55],
    "macd_fast": [5, 8, 10, 12],
    "macd_slow": [15, 18, 21, 26],
    "macd_signal": [5, 7, 9], 
    "drawdown_window": list(range(100, 251, 50)),
    "drawdown_no_rain": [-0.05, -0.08, -0.10],
    "adx_period": [7, 10, 14],
    "adx_threshold": [15, 20, 25],
    "bb_window": [15, 20],
    "bb_std_dev": [2, 2.5]
}

# --- 核心回測功能函數 ---
def run_single_backtest(df, config, strategy_type='conservative'):
    # 使用從 core.scan_module 導入的函式
    df_calc = calculate_indicators(df.copy(), config)
    df_calc['Barometer_Status'] = df_calc.apply(get_barometer_status, axis=1, config=config)
    
    recovery_statuses = []
    for i in range(len(df_calc)):
        status = get_recovery_status(df_calc.iloc[i], df_calc.iloc[i-1] if i > 0 else None, config)
        recovery_statuses.append(status)
    df_calc['Recovery_Status'] = recovery_statuses
    
    required_cols = ['ma_long', 'drawdown_window', 'adx_period', 'bb_window']
    min_data_length = max(config.get(key, 0) for key in required_cols)

    df_calc = df_calc.dropna()
    if len(df_calc) < min_data_length: return None

    b_h_return = (df_calc['Close'].iloc[-1] / df_calc['Close'].iloc[0]) - 1

    capital = 1.0
    position, trades, win_trades = 0, 0, 0
    buy_price = 0

    for i in range(1, len(df_calc)):
        row = df_calc.iloc[i]
        
        buy_condition = row['Recovery_Status'] == '撥雲見日'
        sell_condition = "雨天" in row['Barometer_Status'] or "颱風天" in row['Barometer_Status']
        
        if buy_condition and position == 0: 
            position = 1
            buy_price = row['Close']
        elif sell_condition and position == 1:
            position = 0
            trades += 1
            profit = (row['Close'] - buy_price) / buy_price
            if profit > 0: win_trades += 1
            capital *= (1 + profit)
    
    if position == 1:
        trades += 1
        profit = (df_calc['Close'].iloc[-1] - buy_price) / buy_price
        if profit > 0: win_trades += 1
        capital *= (1 + profit)
        
    strategy_return = capital - 1.0
    win_rate = (win_trades / trades) if trades > 0 else 0
    return {"b&h_return": b_h_return, "strategy_return": strategy_return, "win_rate": win_rate, "trades": trades}

def evaluate_config(config, tickers, strategy_type):
    results = []
    for ticker in tickers:
        # 使用從 core.scan_module 導入的 get_stock_data
        _, df = get_stock_data(ticker)
        
        required_cols = ['ma_long', 'drawdown_window', 'adx_period', 'bb_window']
        min_data_length = max(config.get(key, 0) for key in required_cols)

        if df is None or len(df) < min_data_length: continue

        time.sleep(0.2) # 降低 yfinance API 請求頻率
        
        result = run_single_backtest(df, config, strategy_type)
        if result: results.append(result)

    if not results: return None
    
    returns = [1 + r['strategy_return'] for r in results]
    win_rates = [r['win_rate'] for r in results]
    
    # 過濾掉 NaN 或 inf
    returns = [r for r in returns if pd.notna(r) and np.isfinite(r)]
    if not returns: return None

    return {
        "avg_geo_return": gmean(returns) - 1 if returns else 0,
        "avg_win_rate": np.mean(win_rates)
    }

# --- 3. 主優化循環 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="策略參數優化器")
    parser.add_argument(
        '--objective', 
        type=str, 
        choices=['max_return', 'high_winrate'], 
        default='max_return',
        help='優化目標: "max_return" (最大化報酬率) 或 "high_winrate" (最大化勝率).'
    )
    parser.add_argument(
        '--strategy_type', 
        type=str, 
        choices=['conservative', 'aggressive'], 
        default='conservative',
        help='策略類型: "conservative" (保守) 或 "aggressive" (積極).'
    )
    args = parser.parse_args()

    if args.strategy_type == 'conservative':
        SELECTED_PARAM_SPACE = PARAM_SPACE_CONSERVATIVE
    else:
        SELECTED_PARAM_SPACE = PARAM_SPACE_AGGRESSIVE

    best_score, best_config, best_details = -float('inf'), None, {}
    print(f"===== 開始參數優化 (目標: {args.objective}, 策略: {args.strategy_type})，執行 {N_TRIALS} 次試驗 ====")
    print(f"測試標的: {', '.join(TICKERS_TO_TEST)}")
    
    for i in range(N_TRIALS):
        trial_config = {key: random.choice(values) for key, values in SELECTED_PARAM_SPACE.items()}
        if trial_config['ma_short'] >= trial_config['ma_long']: continue
            
        print(f"\n--- 試驗 [{i+1}/{N_TRIALS}] ---", end="")
        try:
            details = evaluate_config(trial_config, TICKERS_TO_TEST, args.strategy_type)
            if details:
                if args.objective == 'max_return':
                    score = details['avg_geo_return']
                else:
                    score = details['avg_win_rate'] * 1000 + details['avg_geo_return']

                print(f" 平均報酬率: {details['avg_geo_return']:.2%}, 平均勝率: {details['avg_win_rate']:.2%}")
                
                if score > best_score:
                    best_score, best_config, best_details = score, trial_config, details
                    print(f"🎉 新的最佳參數組合被發現！")
        except Exception as e:
            print(f"試驗時發生錯誤: {e}")
            
    print("\n\n" + "="*40)
    print(f"      🎉 優化完成！(目標: {args.objective}, 策略: {args.strategy_type})")
    print("="*40)
    if best_config:
        print(f"綜合幾何平均報酬率: {best_details.get('avg_geo_return', 0):.2%}")
        print(f"綜合平均勝率: {best_details.get('avg_win_rate', 0):.2%}")
        print("\n最佳參數設定:")
        print("{")
        for key, value in best_config.items():
            print(f"    \"{key}\": {value},")
        print("}")
    else:
        print("未能在本次優化中找到有效的參數組合。")
