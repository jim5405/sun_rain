# Model_conf_finetuned.py
# 優化目標: 最大化夏普比率 (max_sharpe)
# 策略類型: conservative (fine-tuned)
# 綜合平均年化報酬率 (經 optimizer.py 在 0050.TW, 006208.TW, VOO, QQQ 上驗證): +13.54%
# 綜合平均夏普比率 (經 optimizer.py 驗證): 0.89
# 綜合平均勝率 (經 optimizer.py 驗證): 67.20%

CONFIG = {
    "adx_period": 14,
    "adx_threshold": 20,
    "bb_std_dev": 2,
    "bb_window": 25,
    "drawdown_no_rain": -0.1,
    "drawdown_window": 300,
    "ma_long": 200,
    "ma_short": 30,
    "macd_fast": 12,
    "macd_signal": 9,
    "macd_slow": 30,
    "rsi_bear_threshold": 45,
    "rsi_bull_threshold": 55,
    "rsi_oversold": 35,
    "rsi_window": 14,
}
