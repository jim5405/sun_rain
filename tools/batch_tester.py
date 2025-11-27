import pandas as pd
import yfinance as yf
import numpy as np
import time
import argparse
import importlib

# --- 要批量測試的股票列表 ---
# 您可以在此處自定義要批量回測的股票清單
TICKERS_TO_TEST = [
    "0050.TW", "006208.TW", "2330.TW", "VOO", "QQQ", "MSFT", "AAPL"
]

import os
# Add parent directory to sys.path to allow importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- 核心功能函數 ---
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="max", auto_adjust=True)
    if df.empty: return None
    time.sleep(0.5) # 增加延遲，避免 API 限流
    return df

def calculate_indicators(df, config):
    df['ma_short'] = df['Close'].rolling(window=config["ma_short"]).mean()
    df['ma_long'] = df['Close'].rolling(window=config["ma_long"]).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=config["rsi_window"]).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=config["rsi_window"]).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    df['MACD'] = df['Close'].ewm(span=config["macd_fast"], adjust=False).mean() - \
                 df['Close'].ewm(span=config["macd_slow"], adjust=False).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=config["macd_signal"], adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    rolling_max = df['Close'].rolling(window=config["drawdown_window"], min_periods=1).max()
    df['Drawdown'] = (df['Close'] / rolling_max) - 1
    return df

def get_barometer_status(row, config):
    price = row['Close']
    ma_short = row['ma_short']
    ma_long = row['ma_long']
    rsi = row['RSI']
    if pd.isna(ma_long): return "資料不足"
    try:
        if price > ma_short > ma_long and rsi > config["rsi_bull_threshold"]:
            return "☀️ 晴天"
        elif price > ma_short and price > ma_long: return "🌥️ 多雲"
        elif ma_long > price > ma_short or (ma_short > price and price > ma_long): return "☁️ 陰天"
        elif ma_short > price and ma_long > price and rsi < config["rsi_bear_threshold"]:
            return "🌧️ 雨天"
        elif ma_short > price and ma_long > price and rsi < config["rsi_oversold"]:
            return "⛈️ 颱風天"
        else: return "☁️ 陰天"
    except (ValueError, TypeError): return "資料不足"

def get_recovery_status(row, prev_row, config):
    if pd.isna(row['MACD_hist']) or pd.isna(row['Drawdown']): return "資料不足"
    prev_drawdown = prev_row['Drawdown'] if prev_row is not None and not pd.isna(prev_row['Drawdown']) else 0
    if row['Drawdown'] <= config["drawdown_no_rain"] and row['Drawdown'] > prev_drawdown and row['MACD_hist'] > 0:
        return "撥雲見日"
    return "無雨"

def run_single_backtest(df, ticker, config, strategy_type='conservative'):
    df_calc = calculate_indicators(df.copy(), config)
    df_calc['Barometer_Status'] = df_calc.apply(get_barometer_status, axis=1, config=config)
    recovery_statuses = []
    for i in range(len(df_calc)):
        status = get_recovery_status(df_calc.iloc[i], df_calc.iloc[i-1] if i > 0 else None, config)
        recovery_statuses.append(status)
    df_calc['Recovery_Status'] = recovery_statuses
    
    df_calc = df_calc.dropna(subset=['ma_long'])
    if len(df_calc) < 2: return None

    b_h_return = (df_calc['Close'].iloc[-1] / df_calc['Close'].iloc[0]) - 1

    capital, position, trades, win_trades, buy_price = 1.0, 0, 0, 0, 0
    for i in range(1, len(df_calc)):
        row = df_calc.iloc[i]
        
        # 買入條件
        if strategy_type == 'conservative':
            buy_condition = row['Recovery_Status'] == '撥雲見日'
        else: # aggressive
            buy_condition = row['Recovery_Status'] == '撥雲見日' and row['Barometer_Status'] not in ['🌧️ 雨天', '⛈️ 颱風天']

        # 賣出條件
        if strategy_type == 'conservative':
            sell_condition = row['Barometer_Status'] in ['🌧️ 雨天', '⛈️ 颱風天']
        else: # aggressive
            sell_condition = row['Barometer_Status'] in ['☁️ 陰天', '🌧️ 雨天', '⛈️ 颱風天']
        
        if buy_condition and position == 0: position = 1; buy_price = row['Close']
        elif sell_condition and position == 1:
            position = 0; trades += 1; profit = (row['Close'] - buy_price) / buy_price
            if profit > 0: win_trades += 1
            capital *= (1 + profit)
    
    if position == 1:
        trades += 1; profit = (df_calc['Close'].iloc[-1] - buy_price) / buy_price
        if profit > 0: win_trades += 1
        capital *= (1 + profit)
        
    strategy_return = capital - 1.0
    win_rate = (win_trades / trades) if trades > 0 else 0
    return {"ticker": ticker, "b&h_return": b_h_return, "strategy_return": strategy_return, "win_rate": win_rate, "trades": trades}

def load_config(model_name):
    try:
        try:
            module = importlib.import_module(f"config.{model_name}")
        except ImportError:
            module = importlib.import_module(model_name)
        return module.CONFIG
    except ImportError:
        print(f"錯誤: 找不到模型設定檔 '{model_name}.py'")
        exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="批量股票策略回測工具")
    parser.add_argument("--model", type=str, default="Model_conf", help="要使用的模型設定檔。 সন")
    parser.add_argument("--strategy_type", type=str, choices=['conservative', 'aggressive'], default='conservative', help="策略類型。")
    args = parser.parse_args()
    
    CONFIG = load_config(args.model)
    
    results = []
    print(f"===== 開始批量回測 (模型: {args.model}, 策略: {args.strategy_type}) =====")
    
    for ticker in TICKERS_TO_TEST:
        print(f"\n--- 正在處理: {ticker} ---")
        try:
            df = get_stock_data(ticker)
            if df is None or len(df) < max(CONFIG['ma_long'], CONFIG['drawdown_window']):
                print(f"警告: {ticker} 數據不足，跳過。")
                continue

            result = run_single_backtest(df, ticker, CONFIG, args.strategy_type)
            if result:
                results.append(result)
                print(f"  > 買入並持有 (B&H) 報酬率: {result['b&h_return']:.2%}")
                print(f"  > 擇時策略報酬率: {result['strategy_return']:.2%}")
                print(f"  > 交易次數: {result['trades']}")
                print(f"  > 策略勝率: {result['win_rate']:.2%}")
                if result['strategy_return'] > result['b&h_return']:
                    print("  > 結果: 🎉 策略勝出")
                else:
                    print("  > 結果: 表現不如 B&H")
            else:
                print(f"  > {ticker} 未能產生有效回測結果。")
        except Exception as e:
            print(f"處理 {ticker} 時發生錯誤: {e}")

    print("\n\n" + "="*25)
    print(f"      批量回測結果總結 ({args.model} / {args.strategy_type})")
    print("="*25)

    if results:
        print(f"{ '股票代碼':<12} | {'B&H 報酬率':>12} | {'策略報酬率':>12} | {'交易次數':>8} | {'策略勝率':>8} | {'結果':>10}")
        print("-" * 80)
        win_count = 0
        for res in results:
            outcome = "🎉 勝出" if res['strategy_return'] > res['b&h_return'] else "落後"
            if outcome == "🎉 勝出": win_count += 1
            print(f"{res['ticker']:<12} | {res['b&h_return']:>12.2%} | {res['strategy_return']:>12.2%} | {res['trades']:>8} | {res['win_rate']:>8.2%} | {outcome:<10}")
        print("\n" + "-" * 80)
        print(f"總結: 在 {len(results)} 檔股票中，本策略有 {win_count} 檔表現優於買入並持有，勝率為 {(win_count/len(results)):.2%}")
    else:
        print("無任何有效回測結果可顯示。")