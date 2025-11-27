
# Model_conf_alt.py
# 優化目標: 最大化綜合勝率 (high_winrate)
# 策略類型: conservative
# 綜合幾何平均報酬率: +53.32%
# 綜合平均勝率: 85.42%

CONFIG = {
    "ma_short": 60,
    "ma_long": 150,
    "rsi_window": 25,
    "rsi_oversold": 40,
    "rsi_bull_threshold": 60,
    "rsi_bear_threshold": 40,
    "macd_fast": 18,
    "macd_slow": 35,
    "macd_signal": 18,
    "drawdown_window": 350,
    "drawdown_no_rain": -0.15,
    "adx_period": 25,
    "adx_threshold": 30,
}
