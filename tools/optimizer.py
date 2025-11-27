import pandas as pd
import yfinance as yf
import numpy as np
import random
import time
import argparse
from scipy.stats.mstats import gmean
import warnings

# --- 優化器設定 ---
N_TRIALS = 100  # 增加試驗次數以獲得更可靠結果
TICKERS_TO_TEST = [
    "0050.TW", "006208.TW", "VOO", "QQQ" # 以指數型ETF為主，尋找更穩定的策略
]

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. 參數搜索空間 (Search Space) ---
# 保守策略的參數空間
PARAM_SPACE_CONSERVATIVE = {
    "ma_short": list(range(40, 91, 10)), # 較長的短期均線
    "ma_long": list(range(150, 251, 30)), # 較長的長期均線
    "rsi_window": [20, 25, 30],
    "rsi_oversold": [30, 35, 40],
    "rsi_bull_threshold": [50, 55, 60],
    "rsi_bear_threshold": [40, 45, 50],
    "macd_fast": [12, 15, 18],
    "macd_slow": [26, 30, 35],
    "macd_signal": [9, 12, 18], 
    "drawdown_window": list(range(250, 351, 50)),
    "drawdown_no_rain": [-0.10, -0.12, -0.15], # 較深的回撤才觸發
    "adx_period": [14, 20, 25], # ADX 週期
    "adx_threshold": [20, 25, 30] # ADX 趨勢強度閾值
}

# 積極策略的參數空間 (縮短指標天數)
PARAM_SPACE_AGGRESSIVE = {
    "ma_short": list(range(20, 61, 10)), # 較短的短期均線
    "ma_long": list(range(80, 181, 20)), # 較短的長期均線
    "rsi_window": [7, 10, 14, 20],
    "rsi_oversold": [20, 25, 30],
    "rsi_bull_threshold": [55, 60, 65],
    "rsi_bear_threshold": [45, 50, 55],
    "macd_fast": [5, 8, 10, 12],
    "macd_slow": [15, 18, 21, 26],
    "macd_signal": [5, 7, 9], 
    "drawdown_window": list(range(100, 251, 50)),
    "drawdown_no_rain": [-0.05, -0.08, -0.10], # 較淺的回撤就觸發
    "adx_period": [7, 10, 14], # ADX 週期
    "adx_threshold": [15, 20, 25] # ADX 趨勢強度閾值
}


# --- 核心回測功能函數 ---
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="max", auto_adjust=True)
    if df.empty: return None
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

    # 計算 ADX
    df['TR'] = np.maximum(np.maximum(df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift(1))), np.abs(df['Low'] - df['Close'].shift(1)))
    df['DMplus'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), np.maximum(df['High'] - df['High'].shift(1), 0), 0)
    df['DMminus'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)
    
    # 平滑處理
    adx_period = config["adx_period"]
    df['TR_exp'] = df['TR'].ewm(span=adx_period, adjust=False).mean()
    df['DMplus_exp'] = df['DMplus'].ewm(span=adx_period, adjust=False).mean()
    df['DMminus_exp'] = df['DMminus'].ewm(span=adx_period, adjust=False).mean()
    
    df['DIplus'] = (df['DMplus_exp'] / df['TR_exp']) * 100
    df['DIminus'] = (df['DMminus_exp'] / df['TR_exp']) * 100
    df['DX'] = np.abs(df['DIplus'] - df['DIminus']) / (df['DIplus'] + df['DIminus']) * 100
    df['ADX'] = df['DX'].ewm(span=adx_period, adjust=False).mean()

    return df

def get_barometer_status(row, config):
    price = row['Close']
    ma_short = row['ma_short']
    ma_long = row['ma_long']
    rsi = row['RSI']
    if pd.isna(ma_long) or pd.isna(ma_short): return "資料不足"
    try:
        if price > ma_short > ma_long and rsi > config["rsi_bull_threshold"]:
            return "☀️ 晴天"
        elif price > ma_short and price > ma_long: return "🌥️ 多雲"
        elif ma_long > price > ma_short or (ma_short > price and price > ma_long): return "☁️ 陰天"
        elif ma_short > price and ma_long > price and rsi < config["rsi_bear_threshold"]:
            return "🌧️ 雨天"
        elif ma_short > price and ma_long > price and rsi < config["rsi_oversold"]: # 使用 rsi_oversold 作為颱風天觸發
            return "⛈️ 颱風天"
        else: return "☁️ 陰天"
    except (ValueError, TypeError): return "資料不足"


def get_recovery_status(row, prev_row, config):
    if pd.isna(row['MACD_hist']) or pd.isna(row['Drawdown']) or pd.isna(row['ADX']) or pd.isna(row['DIplus']) or pd.isna(row['DIminus']): return "資料不足" # ADX 加入判斷
    prev_drawdown = prev_row['Drawdown'] if prev_row is not None and not pd.isna(prev_row['Drawdown']) else 0

    # 撥雲見日訊號 (加入ADX確認趨勢強度)
    if row['Drawdown'] <= config["drawdown_no_rain"] and \
       row['Drawdown'] > prev_drawdown and \
       row['MACD_hist'] > 0 and \
       row['ADX'] > config["adx_threshold"] and \
       row['DIplus'] > row['DIminus']: # 確認ADX趨勢向上
        return "撥雲見日"
    return "無雨"

def run_single_backtest(df, config, strategy_type='conservative'):
    df_calc = calculate_indicators(df.copy(), config)
    df_calc['Barometer_Status'] = df_calc.apply(get_barometer_status, axis=1, config=config)
    
    recovery_statuses = []
    for i in range(len(df_calc)):
        status = get_recovery_status(df_calc.iloc[i], df_calc.iloc[i-1] if i > 0 else None, config)
        recovery_statuses.append(status)
    df_calc['Recovery_Status'] = recovery_statuses
    
    # 確保回測的數據長度足夠，避免因NaN值過多導致無效交易
    df_calc = df_calc.dropna(subset=['ma_short', 'ma_long', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'Drawdown', 'ADX', 'DIplus', 'DIminus', 'Barometer_Status', 'Recovery_Status'])
    if len(df_calc) < max(config['ma_long'], config['drawdown_window'], config['adx_period']): return None # 確保有足夠的數據量進行回測

    b_h_return = (df_calc['Close'].iloc[-1] / df_calc['Close'].iloc[0]) - 1

    capital, initial_capital = 1.0, 1.0
    position, trades, win_trades = 0, 0, 0
    buy_price = 0

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
        
        if buy_condition and position == 0: 
            position = 1; buy_price = row['Close']
        elif sell_condition and position == 1:
            position = 0; trades += 1
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
        df = get_stock_data(ticker)
        # 確保數據長度至少能計算最長的MA和Drawdown_window以及ADX_period
        if df is None or len(df) < max(config['ma_long'], config['drawdown_window'], config['adx_period']): continue
        
        # 每次獲取數據後增加延遲
        time.sleep(0.5) 
        
        result = run_single_backtest(df, config, strategy_type)
        if result: results.append(result)
    if not results: return None
    
    returns = [1 + r['strategy_return'] for r in results]
    win_rates = [r['win_rate'] for r in results]
    
    return {
        "avg_geo_return": gmean(returns) - 1,
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

    # 根據策略類型選擇參數空間
    if args.strategy_type == 'conservative':
        SELECTED_PARAM_SPACE = PARAM_SPACE_CONSERVATIVE
    else: # aggressive
        SELECTED_PARAM_SPACE = PARAM_SPACE_AGGRESSIVE

    best_score, best_config, best_details = -float('inf'), None, {}
    print(f"===== 開始參數優化 (目標: {args.objective}, 策略: {args.strategy_type})，執行 {N_TRIALS} 次試驗 ====")
    print(f"測試標的: {', '.join(TICKERS_TO_TEST)}")
    
    for i in range(N_TRIALS):
        trial_config = {key: random.choice(values) for key, values in SELECTED_PARAM_SPACE.items()}
        # 確保 ma_short < ma_long
        if trial_config['ma_short'] >= trial_config['ma_long']: continue
            
        print(f"\n--- 試驗 [{i+1}/{N_TRIALS}] ---", end="")
        try:
            details = evaluate_config(trial_config, TICKERS_TO_TEST, args.strategy_type)
            if details:
                # 根據優化目標計算分數
                if args.objective == 'max_return':
                    score = details['avg_geo_return']
                else: # high_winrate
                    score = details['avg_win_rate'] * 1000 + details['avg_geo_return'] # 高勝率賦予更高權重

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