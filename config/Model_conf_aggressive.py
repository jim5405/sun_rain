# Model_conf_aggressive.py
# 優化目標: 最大化綜合報酬率 (max_return)
# 策略類型: aggressive
# 綜合幾何平均報酬率: +112.47%
# 綜合平均勝率: 47.30%

CONFIG = {
    "ma_short": 50,
    "ma_long": 100,
    "rsi_window": 7,
    "rsi_oversold": 30,
    "rsi_bull_threshold": 60,
    "rsi_bear_threshold": 45,
    "macd_fast": 12,
    "macd_slow": 18,
    "macd_signal": 7,
    "drawdown_window": 250,
    "drawdown_no_rain": -0.05,
    "adx_period": 10,
    "adx_threshold": 20,
}