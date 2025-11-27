# -*- coding: utf-8 -*-
# scan_module.py

import pandas as pd
import yfinance as yf
import numpy as np
import time
import importlib
import os
import sys

# --- Hold List File ---
HOLD_LIST_FILE = "hold_list.txt"

# --- Stock List Acquisition Functions ---
def _get_sp500_tickers_from_wiki():
    """Use a curated list of major US ETFs/stocks, avoiding direct Wikipedia scraping issues."""
    return ["VOO", "QQQ", "SPY", "DIA", "IWM", "AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "TSLA", "META"]

def _get_twse_comprehensive_list():
    """
    Returns a comprehensive curated list of major TWSE and TPEx stocks.
    This replaces the smaller list to fulfill the broad scanning requirement.
    """
    return [
        # Taiwan 50
        "2330.TW", "2454.TW", "2317.TW", "2881.TW", "2308.TW", "2882.TW", "2886.TW", "2303.TW",
        "2891.TW", "1301.TW", "1303.TW", "2002.TW", "1216.TW", "2884.TW", "3711.TW", "2207.TW",
        "2885.TW", "6505.TW", "3034.TW", "3037.TW", "5880.TW", "2395.TW", "2892.TW", "1101.TW",
        "2880.TW", "2883.TW", "2301.TW", "2912.TW", "2357.TW", "2412.TW", "3008.TW", "2382.TW",
        "5871.TW", "2379.TW", "1326.TW", "3045.TW", "1605.TW", "2887.TW", "2327.TW", "2890.TW",
        "4904.TW", "6669.TW", "2345.TW", "1590.TW", "1102.TW", "9910.TW", "2408.TW", "2474.TW",
        # ETFs
        "0050.TW", "0056.TW", "00878.TW", "006208.TW",
        # Mid-Cap 100 (selection)
        "2603.TW", "2609.TW", "2615.TW", "2006.TW", "2014.TW", "1722.TW", "1795.TW", "2344.TW",
        "2377.TW", "2409.TW", "3044.TW", "3231.TW", "3533.TW", "3661.TW", "4958.TW",
        "5269.TW", "6176.TW", "6239.TW", "6278.TW", "8069.TWO", "8299.TWO", "9921.TW",
        # TPEx (OTC, selection)
        "6488.TWO", "8044.TWO", "6147.TWO", "5347.TWO", "4979.TWO", "3293.TWO", "6245.TWO", "3105.TWO",
        "5483.TWO", "4129.TWO", "8436.TWO", "6182.TWO", "3264.TWO", "6121.TWO", "5274.TWO"
    ]

def get_dynamic_scan_list():
    """Dynamically generate scan list, combining US stocks and a comprehensive TWSE list."""
    sp500_tickers = _get_sp500_tickers_from_wiki()
    twse_tickers = _get_twse_comprehensive_list()
    all_tickers = list(set(sp500_tickers + twse_tickers))
    return all_tickers

def read_hold_list():
    if not os.path.exists(HOLD_LIST_FILE): return set()
    with open(HOLD_LIST_FILE, 'r', encoding='utf-8') as f:
        return {line.strip().upper() for line in f if line.strip()}

def load_config(model_name):
    try:
        module = importlib.import_module(model_name)
        return module.CONFIG
    except ImportError:
        print(f"Error: Model config file '{model_name}.py' not found.", file=sys.stderr)
        exit(1)

# --- Core Analysis Functions ---
# Note: These functions are now self-contained in this module
def get_stock_data(ticker, period="2y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, auto_adjust=True)
    time.sleep(0.1) # Add a small delay to be kind to the yfinance API
    if df.empty: return ticker, None
    return ticker, df

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
    # ADX calculation
    adx_period = config.get("adx_period", 14) # Use default if not in config
    df['TR'] = np.maximum(np.maximum(df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift(1))), np.abs(df['Low'] - df['Close'].shift(1)))
    df['DMplus'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), np.maximum(df['High'] - df['High'].shift(1), 0), 0)
    df['DMminus'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)
    df['TR_exp'] = df['TR'].ewm(span=adx_period, adjust=False).mean()
    df['DMplus_exp'] = df['DMplus'].ewm(span=adx_period, adjust=False).mean()
    df['DMminus_exp'] = df['DMminus'].ewm(span=adx_period, adjust=False).mean()
    df['DIplus'] = (df['DMplus_exp'] / df['TR_exp']).replace(np.inf, 0).fillna(0) * 100
    df['DIminus'] = (df['DMminus_exp'] / df['TR_exp']).replace(np.inf, 0).fillna(0) * 100
    df['DX'] = np.abs(df['DIplus'] - df['DIminus']) / (df['DIplus'] + df['DIminus']).replace(0, np.nan) * 100
    df['ADX'] = df['DX'].ewm(span=adx_period, adjust=False).mean()
    return df

def get_barometer_status(row, config):
    price = row['Close']; ma_short = row['ma_short']; ma_long = row['ma_long']; rsi = row['RSI']
    if pd.isna(ma_long) or pd.isna(ma_short): return "數據不足"
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
    except (ValueError, TypeError): return "數據不足"

def get_recovery_status(row, prev_row, config):
    if pd.isna(row['MACD_hist']) or pd.isna(row['Drawdown']) or pd.isna(row.get('ADX')): return "數據不足"
    prev_drawdown = prev_row['Drawdown'] if prev_row is not None and not pd.isna(prev_row['Drawdown']) else 0
    adx_threshold = config.get("adx_threshold", 20) # Use default if not in config
    if row['Drawdown'] <= config["drawdown_no_rain"] and row['Drawdown'] > prev_drawdown and row['MACD_hist'] > 0 and row['ADX'] > adx_threshold and row['DIplus'] > row['DIminus']:
        return "撥雲見日"
    return "無雨"

def get_recommendation_verbal(barometer, recovery):
    if recovery == "撥雲見日": return "🟢 建議進場"
    if "雨天" in barometer or "颱風天" in barometer: return "🔴 建議出場或空手"
    return "🟡 建議持有或觀望"

def get_recommendation_score(barometer, recovery):
    if recovery == "撥雲見日": return 1
    if "雨天" in barometer or "颱風天" in barometer: return -1
    return 0

def get_combined_recommendation_score(s1, s2): return s1 + s2

def get_final_verbal_score(cs):
    if cs == 2: return "💎 強力買入"
    elif cs == 1: return "🟢 建議買入"
    elif cs == 0: return "🟡 持有/觀望"
    elif cs == -1: return "🟠 建議減碼"
    elif cs == -2: return "🔴 強力賣出"
    else: return "❓ 未知狀態"

def analyze_ticker(ticker, config, strategy_type='conservative'):
    try:
        _, df = get_stock_data(ticker)
        required_len = max(config.get('ma_long', 250), config.get('drawdown_window', 250))
        if df is None or len(df) < required_len:
            return ticker, None, "數據不足"
        
        df = calculate_indicators(df, config)
        df_valid = df.dropna(subset=['ma_long', 'ADX'])
        if len(df_valid) < 2: return ticker, None, "指標計算後數據不足"

        last_row, prev_row = df_valid.iloc[-1], df_valid.iloc[-2]
        
        barometer = get_barometer_status(last_row, config)
        recovery = get_recovery_status(last_row, prev_row, config)
        recommendation = get_recommendation_verbal(barometer, recovery)
        
        result = {
            "ticker": ticker, "price": f"{last_row['Close']:.2f}", "barometer": barometer,
            "recovery": recovery, "recommendation": recommendation, "date": last_row.name.strftime('%Y-%m-%d')
        }
        return ticker, result, "成功"
    except Exception as e:
        return ticker, None, f"錯誤: {e}"