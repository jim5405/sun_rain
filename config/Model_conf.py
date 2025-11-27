# Model_conf.py
# 優化目標: 最大化綜合報酬率 (max_return)
# 策略類型: conservative
# 綜合幾何平均報酬率 (經 optimizer.py 在 0050.TW, 006208.TW, VOO, QQQ 上驗證): +23.86%
# 綜合平均勝率 (經 optimizer.py 驗證): 84.38%

CONFIG = {
    "ma_short": 90,
    "ma_long": 180,
    "rsi_window": 20,
    "rsi_oversold": 35,
    "rsi_bull_threshold": 55,
    "rsi_bear_threshold": 40,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 18,
    "drawdown_window": 300,
    "drawdown_no_rain": -0.1,
    "adx_period": 14,
    "adx_threshold": 20,
    "bb_window": 25,
    "bb_std_dev": 2,
}
