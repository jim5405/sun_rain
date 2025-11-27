import unittest
import pandas as pd
import numpy as np
import importlib

# --- 從 core.scan_module 導入最新的核心函數 ---
from core.scan_module import (
    calculate_indicators,
    get_barometer_status,
    get_recovery_status,
    get_recommendation_verbal as get_recommendation # 保持舊的測試命名
)

# 載入 Model_conf 作為測試用配置
try:
    from config import Model_conf
    TEST_CONFIG = Model_conf.CONFIG
except ImportError:
    print("錯誤: 找不到 config/Model_conf.py，將使用預設配置。\n")
    TEST_CONFIG = {
        "ma_short": 50, "ma_long": 200, "rsi_window": 14, "rsi_oversold": 30,
        "rsi_bull_threshold": 50, "rsi_bear_threshold": 40, "macd_fast": 12,
        "macd_slow": 26, "macd_signal": 9, "drawdown_window": 250,
        "drawdown_no_rain": -0.1, "adx_period": 14, "adx_threshold": 20,
        "bb_window": 20, "bb_std_dev": 2 # 加入布林通道的預設值
    }

# --- 模擬的回測函數 (因為原始的回測函數不在 scan_module 中) ---
# 這個函數是為了測試目的而保留在 test_suite.py 中的
def run_single_backtest(df, ticker, config, strategy_type='conservative'):
    # 注意：為了讓測試更容易，我們假設傳入的 df 已經計算好所有指標
    df_calc = df.copy()
    
    # 確保所有需要的欄位都存在
    required_cols = ['ma_short', 'ma_long', 'RSI', 'MACD_hist', 'Drawdown', 'ADX', 'DIplus', 'DIminus', 'bb_upper', 'bb_lower', 'Recovery_Status', 'Barometer_Status']
    df_calc = df_calc.dropna(subset=required_cols)
    if len(df_calc) < 2: return None

    b_h_return = (df_calc['Close'].iloc[-1] / df_calc['Close'].iloc[0]) - 1

    capital, initial_capital = 1.0, 1.0
    position, trades, win_trades = 0, 0, 0
    buy_price = 0

    for i in range(1, len(df_calc)):
        row = df_calc.iloc[i]
        
        buy_condition = row['Recovery_Status'] == '撥雲見日'

        # 更新賣出條件以匹配 scan_module 的邏輯
        is_rainy = "雨天" in row['Barometer_Status'] or "颱風天" in row['Barometer_Status']
        sell_condition = is_rainy
        
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


class TestTradingStrategy(unittest.TestCase):

    def setUp(self):
        self.mock_df_length = 350
        dates = pd.date_range(start='2023-01-01', periods=self.mock_df_length, freq='D')
        self.mock_df = pd.DataFrame({
            'High': np.random.rand(self.mock_df_length) * 10 + 100,
            'Low': np.random.rand(self.mock_df_length) * 10 + 90,
            'Close': np.linspace(95, 105, self.mock_df_length) + np.random.randn(self.mock_df_length) * 2,
            'Volume': np.random.randint(100000, 1000000, self.mock_df_length)
        }, index=dates)
        self.mock_df['High'] = self.mock_df['Close'] + np.random.rand(self.mock_df_length) * 2
        self.mock_df['Low'] = self.mock_df['Close'] - np.random.rand(self.mock_df_length) * 2
        self.config = TEST_CONFIG

    def test_calculate_indicators(self):
        df_indicators = calculate_indicators(self.mock_df.copy(), self.config)
        self.assertIn('ma_short', df_indicators.columns)
        self.assertIn('ADX', df_indicators.columns)
        self.assertIn('bb_upper', df_indicators.columns) # 檢查布林通道
        self.assertGreater(df_indicators['bb_upper'].count(), self.mock_df_length - self.config['bb_window'] - 1)
        self.assertFalse(np.isinf(df_indicators['ADX']).any())

    def test_get_barometer_status_sunny_rainy(self):
        test_df = self.mock_df.copy().tail(5).reset_index(drop=True)
        # 晴天測試
        test_df['Close'] = [100, 105, 110, 115, 120]
        test_df['ma_short'] = [90, 95, 100, 105, 110]
        test_df['ma_long'] = [80, 85, 90, 95, 100]
        test_df['RSI'] = [70, 75, 80, 85, 90]
        test_df['bb_upper'] = [118, 119, 120, 121, 122] # 價格低於上軌
        test_df['bb_lower'] = [80, 85, 90, 95, 100]
        self.assertEqual(get_barometer_status(test_df.iloc[-1], self.config), "☀️ 晴天")
        
        # 晴天過熱測試
        test_df.loc[test_df.index[-1], 'Close'] = 123 # 價格高於上軌
        self.assertEqual(get_barometer_status(test_df.iloc[-1], self.config), "☀️ 晴天 (注意過熱)")

        # 雨天測試
        test_df['Close'] = [120, 115, 110, 105, 100]
        test_df['ma_short'] = [130, 125, 120, 115, 110]
        test_df['ma_long'] = [140, 135, 130, 125, 120]
        test_df['RSI'] = [30, 25, 20, 15, 10]
        test_df['bb_upper'] = [140, 135, 130, 125, 120]
        test_df['bb_lower'] = [102, 101, 100, 99, 98] # 價格高於下軌
        self.assertEqual(get_barometer_status(test_df.iloc[-1], self.config), "⛈️ 颱風天")

        # 颱風天恐慌測試
        test_df.loc[test_df.index[-1], 'Close'] = 97 # 價格低於下軌
        self.assertEqual(get_barometer_status(test_df.iloc[-1], self.config), "⛈️ 颱風天 (注意恐慌)")

    def test_get_recovery_status_buy_signal(self):
        test_df = self.mock_df.copy().tail(5).reset_index(drop=True)
        test_df['Drawdown'] = [-0.15, -0.15, -0.13, -0.11, -0.09]
        test_df['MACD_hist'] = [-0.5, -0.2, 0.1, 0.3, 0.5]
        test_df['ADX'] = [15, 20, 25, 30, 35]
        test_df['DIplus'] = [20, 25, 30, 35, 40]
        test_df['DIminus'] = [40, 35, 30, 25, 20]
        config_for_test = self.config.copy()
        config_for_test['drawdown_no_rain'] = -0.08
        config_for_test['adx_threshold'] = 25
        self.assertEqual(get_recovery_status(test_df.iloc[-1], test_df.iloc[-2], config_for_test), "撥雲見日")

        test_df.loc[test_df.index[-1], 'ADX'] = 10
        self.assertEqual(get_recovery_status(test_df.iloc[-1], test_df.iloc[-2], config_for_test), "無雨")

    def test_get_recommendation(self):
        self.assertEqual(get_recommendation("☁️ 陰天", "撥雲見日"), "🟢 建議進場")
        self.assertEqual(get_recommendation("🌧️ 雨天", "無雨"), "🔴 建議出場或空手")
        self.assertEqual(get_recommendation("🌥️ 多雲", "無雨"), "🟡 建議持有或觀望")
        
    def test_run_single_backtest_logic(self):
        # 此測試旨在驗證回測的基本買賣邏輯
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        test_df = pd.DataFrame({'Close': np.linspace(100, 110, 10)}, index=dates)
        
        # 手動填充所有需要的欄位以模擬場景
        # 讓所有指標都有效
        for col in ['High', 'Low', 'ma_short', 'ma_long', 'RSI', 'MACD', 'MACD_hist', 'MACD_signal', 'Drawdown', 'ADX', 'DIplus', 'DIminus', 'bb_ma', 'bb_std', 'bb_upper', 'bb_lower']:
            test_df[col] = 1.0

        # 設定一個明確的買賣點
        # Day 3: Buy signal
        # Day 6: Sell signal
        test_df['Recovery_Status'] = ['無雨'] * 10
        test_df.loc[test_df.index[2], 'Recovery_Status'] = '撥雲見日'
        
        test_df['Barometer_Status'] = ['☀️ 晴天'] * 10
        test_df.loc[test_df.index[5], 'Barometer_Status'] = '🌧️ 雨天'

        result = run_single_backtest(test_df, "TEST", self.config)
        self.assertIsNotNone(result)

        buy_price = test_df.iloc[2]['Close']
        sell_price = test_df.iloc[5]['Close']
        expected_profit = (sell_price / buy_price) - 1

        self.assertEqual(result['trades'], 1)
        self.assertAlmostEqual(result['strategy_return'], expected_profit, places=4)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
