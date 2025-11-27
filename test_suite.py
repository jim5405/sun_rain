import unittest
import pandas as pd
import numpy as np
import importlib

# --- 從 judge_stock.py (或其他核心腳本) 複製核心函數 ---
# 確保這些函數是最新且正確的版本
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
    
    adx_period = config["adx_period"]
    df['TR_exp'] = df['TR'].ewm(span=adx_period, adjust=False).mean()
    df['DMplus_exp'] = df['DMplus'].ewm(span=adx_period, adjust=False).mean()
    df['DMminus_exp'] = df['DMminus'].ewm(span=adx_period, adjust=False).mean()
    
    # 避免除以零，TR_exp為0時DIplus/minus為0
    df['DIplus'] = (df['DMplus_exp'] / df['TR_exp']).replace(np.inf, 0).fillna(0) * 100
    df['DIminus'] = (df['DMminus_exp'] / df['TR_exp']).replace(np.inf, 0).fillna(0) * 100
    
    df['DX'] = np.abs(df['DIplus'] - df['DIminus']) / (df['DIplus'] + df['DIminus']).replace(0, np.nan) * 100
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
        elif ma_short > price and ma_long > price and rsi < config["rsi_oversold"]:
            return "⛈️ 颱風天"
        else: return "☁️ 陰天"
    except (ValueError, TypeError): return "資料不足"


def get_recovery_status(row, prev_row, config):
    if pd.isna(row['MACD_hist']) or pd.isna(row['Drawdown']) or pd.isna(row['ADX']) or pd.isna(row['DIplus']) or pd.isna(row['DIminus']): return "資料不足"
    prev_drawdown = prev_row['Drawdown'] if prev_row is not None and not pd.isna(prev_row['Drawdown']) else 0

    if row['Drawdown'] <= config["drawdown_no_rain"] and \
       row['Drawdown'] > prev_drawdown and \
       row['MACD_hist'] > 0 and \
       row['ADX'] > config["adx_threshold"] and \
       row['DIplus'] > row['DIminus']:
        return "撥雲見日"
    return "無雨"

def get_recommendation(barometer, recovery):
    if recovery == "撥雲見日":
        return "🟢 建議進場"
    if "雨天" in barometer or "颱風天" in barometer:
        return "🔴 建議出場或空手"
    return "🟡 建議持有或觀望"


# 修正 run_single_backtest 的函數簽名，加入 ticker 和 strategy_type
def run_single_backtest(df, ticker, config, strategy_type='conservative'):
    df_calc = calculate_indicators(df.copy(), config)
    df_calc['Barometer_Status'] = df_calc.apply(get_barometer_status, axis=1, config=config)
    
    recovery_statuses = []
    for i in range(len(df_calc)):
        status = get_recovery_status(df_calc.iloc[i], df_calc.iloc[i-1] if i > 0 else None, config)
        recovery_statuses.append(status)
    df_calc['Recovery_Status'] = recovery_statuses
    
    df_calc = df_calc.dropna(subset=['ma_short', 'ma_long', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'Drawdown', 'ADX', 'DIplus', 'DIminus', 'Barometer_Status', 'Recovery_Status'])
    if len(df_calc) < max(config.get('ma_long', 0), config.get('drawdown_window', 0), config.get('adx_period', 0)): return None # 確保有足夠的數據量進行回測

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


# 載入 Model_conf 作為測試用配置
try:
    Model_conf = importlib.import_module("Model_conf")
    TEST_CONFIG = Model_conf.CONFIG
except ImportError:
    print("錯誤: 找不到 Model_conf.py，將使用預設配置。\n")
    TEST_CONFIG = { # 設置一個預設的配置以避免程式崩潰
        "ma_short": 50, "ma_long": 200, "rsi_window": 14, "rsi_oversold": 30,
        "rsi_bull_threshold": 50, "rsi_bear_threshold": 40, "macd_fast": 12,
        "macd_slow": 26, "macd_signal": 9, "drawdown_window": 250,
        "drawdown_no_rain": -0.1, "adx_period": 14, "adx_threshold": 20
    }


class TestTradingStrategy(unittest.TestCase):

    def setUp(self):
        # 建立一個模擬的股票數據，足以計算所有指標
        # 確保數據長度足以讓所有指標都有值 (ma_long=240, drawdown_window=300, adx_period=14)
        # 所以至少需要 300+ 筆數據
        self.mock_df_length = 350 # 增加長度以確保dropna後仍有數據
        dates = pd.date_range(start='2023-01-01', periods=self.mock_df_length, freq='D')
        self.mock_df = pd.DataFrame({
            'High': np.random.rand(self.mock_df_length) * 10 + 100,
            'Low': np.random.rand(self.mock_df_length) * 10 + 90,
            'Close': np.random.rand(self.mock_df_length) * 10 + 95,
            'Volume': np.random.randint(100000, 1000000, self.mock_df_length)
        }, index=dates)
        # 調整 Close 數據以創建特定的趨勢
        self.mock_df['Close'] = np.linspace(95, 105, self.mock_df_length) + np.random.randn(self.mock_df_length) * 2
        self.mock_df['High'] = self.mock_df['Close'] + np.random.rand(self.mock_df_length) * 2
        self.mock_df['Low'] = self.mock_df['Close'] - np.random.rand(self.mock_df_length) * 2

        # 使用一個已知的配置
        self.config = TEST_CONFIG

    def test_calculate_indicators(self):
        df_indicators = calculate_indicators(self.mock_df.copy(), self.config)
        # 檢查關鍵指標列是否存在且沒有過多的 NaN
        self.assertIn('ma_short', df_indicators.columns)
        self.assertIn('ma_long', df_indicators.columns)
        self.assertIn('RSI', df_indicators.columns)
        self.assertIn('MACD', df_indicators.columns)
        self.assertIn('MACD_hist', df_indicators.columns)
        self.assertIn('Drawdown', df_indicators.columns)
        self.assertIn('ADX', df_indicators.columns)
        self.assertIn('DIplus', df_indicators.columns)
        self.assertIn('DIminus', df_indicators.columns)
        self.assertGreater(df_indicators['ma_short'].count(), self.mock_df_length - self.config['ma_short'] - 1)
        self.assertGreater(df_indicators['ADX'].count(), self.mock_df_length - self.config['adx_period'] - 1)
        # 驗證 ADX 不會出現除以零的錯誤 (NaN 以外)
        self.assertFalse(np.isinf(df_indicators['ADX']).any(), "ADX 計算結果不應出現 Inf 值")
        
    def test_get_barometer_status_sunny_rainy(self):
        # 測試晴天條件: 股價在所有均線之上, RSI高
        test_df = self.mock_df.copy().tail(5).reset_index(drop=True)
        test_df['Close'] = [100, 105, 110, 115, 120]
        test_df['ma_short'] = [90, 95, 100, 105, 110]
        test_df['ma_long'] = [80, 85, 90, 95, 100]
        test_df['RSI'] = [70, 75, 80, 85, 90]
        
        row_sunny = test_df.iloc[-1]
        self.assertEqual(get_barometer_status(row_sunny, self.config), "☀️ 晴天")

        # 測試雨天條件: 股價在所有均線之下, RSI低
        test_df['Close'] = [120, 115, 110, 105, 100]
        test_df['ma_short'] = [130, 125, 120, 115, 110]
        test_df['ma_long'] = [140, 135, 130, 125, 120]
        test_df['RSI'] = [30, 25, 20, 15, 10]

        row_rainy = test_df.iloc[-1]
        self.assertEqual(get_barometer_status(row_rainy, self.config), "🌧️ 雨天")
        
    def test_get_recovery_status_buy_signal(self):
        # 建立數據以觸發撥雲見日
        # 要求: Drawdown <= drawdown_no_rain, Drawdown > prev_drawdown, MACD_hist > 0, ADX > adx_threshold, DIplus > DIminus
        test_df = self.mock_df.copy().tail(5).reset_index(drop=True)
        # 調整數據以符合條件
        test_df['Drawdown'] = [-0.15, -0.15, -0.13, -0.11, -0.09] # 回撤從深變淺
        test_df['MACD_hist'] = [-0.5, -0.2, 0.1, 0.3, 0.5] # MACD 轉正
        test_df['ADX'] = [15, 20, 25, 30, 35] # ADX 高於閾值
        test_df['DIplus'] = [20, 25, 30, 35, 40]
        test_df['DIminus'] = [40, 35, 30, 25, 20] # DIplus > DIminus

        # 確保 prev_row 存在且 Drawdown 符合條件
        prev_row = test_df.iloc[-2]
        current_row = test_df.iloc[-1]
        
        # 由於 get_recovery_status 簡化, 參數會直接從 config 讀取
        config_for_test = self.config.copy()
        config_for_test['drawdown_no_rain'] = -0.08 # 調整此值以適應 -0.09
        config_for_test['adx_threshold'] = 25
        
        self.assertEqual(get_recovery_status(current_row, prev_row, config_for_test), "撥雲見日")
        
        # 測試不滿足條件的情況 (例如 ADX 不足)
        current_row['ADX'] = 10 # 低於閾值
        self.assertEqual(get_recovery_status(current_row, prev_row, config_for_test), "無雨")


    def test_get_recommendation(self):
        self.assertEqual(get_recommendation("☁️ 陰天", "撥雲見日"), "🟢 建議進場")
        self.assertEqual(get_recommendation("☀️ 晴天", "撥雲見日"), "🟢 建議進場")
        self.assertEqual(get_recommendation("🌧️ 雨天", "無雨"), "🔴 建議出場或空手")
        self.assertEqual(get_recommendation("⛈️ 颱風天", "無雨"), "🔴 建議出場或空手")
        self.assertEqual(get_recommendation("🌥️ 多雲", "無雨"), "🟡 建議持有或觀望")
        self.assertEqual(get_recommendation("☁️ 陰天", "無雨"), "🟡 建議持有或觀望")
        
    def test_run_single_backtest_conservative_aggressive(self):
        # 建立一個非常簡單的數據以驗證買賣邏輯
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D') # 增加數據長度
        test_df = pd.DataFrame({
            'Open': np.linspace(95, 105, 50),
            'High': np.linspace(100, 110, 50),
            'Low': np.linspace(90, 100, 50),
            'Close': np.linspace(98, 108, 50),
            'Volume': [100000]*50
        }, index=dates)
        
        # 填充指標所需的所有欄位，並確保無 NaN
        test_df['ma_short'] = test_df['Close'].rolling(window=5, min_periods=1).mean()
        test_df['ma_long'] = test_df['Close'].rolling(window=10, min_periods=1).mean()
        test_df['RSI'] = np.linspace(40, 60, 50)
        test_df['MACD'] = np.linspace(-1, 1, 50)
        test_df['MACD_hist'] = np.linspace(-0.5, 0.5, 50)
        test_df['MACD_signal'] = np.linspace(-0.8, 0.8, 50)
        test_df['Drawdown'] = np.linspace(-0.2, 0, 50)
        test_df['ADX'] = np.linspace(10, 40, 50)
        test_df['DIplus'] = np.linspace(10, 40, 50)
        test_df['DIminus'] = np.linspace(40, 10, 50)


        # 硬塞訊號以測試邏輯
        barometer_statuses = ['☀️ 晴天'] * 50
        barometer_statuses[4] = '☁️ 陰天' # Day 5 aggressive sell
        barometer_statuses[5] = '🌧️ 雨天' # Day 6 conservative sell
        test_df['Barometer_Status'] = barometer_statuses

        recovery_statuses = ['無雨'] * 50
        recovery_statuses[2] = '撥雲見日' # Day 3 buy
        test_df['Recovery_Status'] = recovery_statuses


        # 簡化配置以適應短數據
        config_for_test = self.config.copy()
        config_for_test['ma_short'] = 1
        config_for_test['ma_long'] = 2
        config_for_test['rsi_window'] = 1
        config_for_test['macd_fast'] = 1
        config_for_test['macd_slow'] = 2
        config_for_test['macd_signal'] = 1
        config_for_test['drawdown_window'] = 2
        config_for_test['adx_period'] = 1


        # 測試保守策略
        # 預期行為: 第三天買入 (test_df.Close[2]), 第六天賣出 (test_df.Close[5], 因為轉雨天)
        # Buy: day 3 (index 2), price test_df.Close[2] (approx 98 + small_noise)
        # Conservative Sell: day 6 (index 5), price test_df.Close[5] (approx 98+noise + 3*~10/50 = 101)
        buy_price_conservative = test_df.iloc[2]['Close']
        sell_price_conservative = test_df.iloc[5]['Close']
        expected_profit_conservative = (sell_price_conservative / buy_price_conservative) - 1
        
        result_conservative = run_single_backtest(test_df.iloc[:7].copy(), "TEST", config_for_test, 'conservative')
        self.assertIsNotNone(result_conservative)
        self.assertAlmostEqual(result_conservative['strategy_return'], expected_profit_conservative, places=2)

        # 測試積極策略
        # 預期行為: 第三天買入 (test_df.Close[2]), 第五天賣出 (test_df.Close[4], 因為轉陰天)
        # Buy: day 3 (index 2), price test_df.Close[2]
        # Aggressive Sell: day 5 (index 4), price test_df.Close[4] (approx 98+noise + 2*~10/50 = 100)
        buy_price_aggressive = test_df.iloc[2]['Close']
        sell_price_aggressive = test_df.iloc[4]['Close']
        expected_profit_aggressive = (sell_price_aggressive / buy_price_aggressive) - 1

        result_aggressive = run_single_backtest(test_df.iloc[:6].copy(), "TEST", config_for_test, 'aggressive')
        self.assertIsNotNone(result_aggressive)
        self.assertAlmostEqual(result_aggressive['strategy_return'], expected_profit_aggressive, places=2)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)