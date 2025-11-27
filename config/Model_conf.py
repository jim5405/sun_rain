
# Model_conf.py
# 優化目標: 最大化綜合報酬率 (max_return)
# 策略類型: conservative
# 綜合幾何平均報酬率: +441.18%
# 綜合平均勝率: 62.12%

CONFIG = {
    "ma_short": 80,
    "ma_long": 240,
    "rsi_window": 30,
    "rsi_oversold": 30,
    "rsi_bull_threshold": 50,
    "rsi_bear_threshold": 40,
    "macd_fast": 12,
    "macd_slow": 30,
    "macd_signal": 18,
    "drawdown_window": 300,
    "drawdown_no_rain": -0.1,
    "adx_period": 14,
    "adx_threshold": 25,
    "bb_window": 20,
    "bb_std_dev": 2,
}
