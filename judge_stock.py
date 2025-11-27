import pandas as pd
import yfinance as yf
import numpy as np
import sys
import argparse
import importlib

# --- 核心功能函數 ---
def get_stock_data(ticker, period="2y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, auto_adjust=True)
    if df.empty:
        raise ValueError(f"無法獲取股票代碼 {ticker} 的數據，請檢查代碼是否正確。")
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
    df['MACD_signal'] = df['MACD'].ewm(span=config["macd_fast"], adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    rolling_max = df['Close'].rolling(window=config["drawdown_window"], min_periods=1).max()
    df['Drawdown'] = (df['Close'] / rolling_max) - 1
    return df

def get_barometer_status(row, config):
    if pd.isna(row['ma_long']) or pd.isna(row['ma_short']): return "資料不足"
    if row['Close'] < row['ma_short'] and row['Close'] < row['ma_long']: return "🌧️ 雨天"
    if row['Close'] > row['ma_short'] > row['ma_long']: return "☀️ 晴天"
    if row['Close'] > row['ma_short'] and row['Close'] > row['ma_long']: return "🌥️ 多雲"
    return "☁️ 陰天"

def get_recovery_status(row, prev_row, config):
    if pd.isna(row['MACD_hist']) or pd.isna(row['Drawdown']): return "資料不足"
    prev_drawdown = prev_row['Drawdown'] if prev_row is not None and not pd.isna(prev_row['Drawdown']) else 0
    if row['Drawdown'] <= config["drawdown_no_rain"] and row['Drawdown'] > prev_drawdown and row['MACD_hist'] > 0:
        return "撥雲見日"
    return "無雨"

def get_recommendation(barometer, recovery):
    if recovery == "撥雲見日":
        return "🟢 建議進場 (偵測到 '撥雲見日' 買入訊號)"
    if "雨天" in barometer:
        return "🔴 建議出場或空手 (市場進入空頭趨勢)"
    return "🟡 建議持有或觀望 (未出現明確的進出場訊號)"

def load_config(model_name):
    """動態導入指定的模型設定檔"""
    try:
        module = importlib.import_module(model_name)
        return module.CONFIG
    except ImportError:
        print(f"錯誤: 找不到模型設定檔 '{model_name}.py'。")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="股票狀態即時診斷工具")
    parser.add_argument("ticker", help="要分析的股票代碼 (例如: 0050.TW 或 AAPL)。")
    parser.add_argument(
        "--model", 
        type=str, 
        default="Model_conf",
        help="要使用的模型設定檔 (預設: Model_conf)。可選: Model_conf_alt"
    )
    args = parser.parse_args()

    TICKER = args.ticker.upper()
    
    try:
        print(f"正在分析股票: {TICKER} (使用模型: {args.model})")
        CONFIG = load_config(args.model)
        
        df = get_stock_data(TICKER)
        df = calculate_indicators(df, CONFIG)
        
        if len(df) < 2: raise ValueError("數據不足，無法進行判斷。")
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        latest_barometer = get_barometer_status(last_row, CONFIG)
        latest_recovery = get_recovery_status(last_row, prev_row, CONFIG)
        recommendation = get_recommendation(latest_barometer, latest_recovery)
        
        print("\n" + "="*40)
        print(f"分析日期: {last_row.name.strftime('%Y-%m-%d')}")
        print(f"當前股價: {last_row['Close']:.2f}")
        print("-" * 40)
        print(f"市場晴雨表: {latest_barometer}")
        print(f"市場放晴指標: {latest_recovery}")
        print("="*40)
        print(f"\n操作建議: {recommendation}\n")

    except Exception as e:
        print(f"\n發生錯誤: {e}")
        sys.exit(1)