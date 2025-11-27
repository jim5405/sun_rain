import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import math

# --- 參數設定區 ---
CONFIGS = {
    "default": {
        "ma_short": 50, "ma_long": 200, "rsi_window": 14, "rsi_overbought": 70,
        "rsi_oversold": 30, "rsi_bull_threshold": 55, "rsi_bear_threshold": 45,
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "drawdown_window": 252,
        "drawdown_no_rain": -0.05, "drawdown_heavy_rain": -0.20
    },
    "robust": {
        "ma_short": 60, "ma_long": 250, "rsi_window": 20, "rsi_overbought": 65,
        "rsi_oversold": 35, "rsi_bull_threshold": 50, "rsi_bear_threshold": 50,
        "macd_fast": 15, "macd_slow": 30, "macd_signal": 12, "drawdown_window": 300,
        "drawdown_no_rain": -0.08, "drawdown_heavy_rain": -0.25
    }
}
# --- 核心功能函數 ---

def get_stock_data(ticker, period="15y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df

def calculate_indicators(df, config):
    df[f'MA{config["ma_short"]}'] = df['Close'].rolling(window=config["ma_short"]).mean()
    df[f'MA{config["ma_long"]}'] = df['Close'].rolling(window=config["ma_long"]).mean()
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
    return df

def get_barometer_status(row, config):
    price = row['Close']
    ma_short = row[f'MA{config["ma_short"]}']
    ma_long = row[f'MA{config["ma_long"]}']
    rsi = row['RSI']
    if pd.isna(ma_long): return "資料不足"
    try:
        if price > ma_short > ma_long and rsi > config["rsi_bull_threshold"]:
            return "晴天"
        elif price > ma_short and price > ma_long: return "多雲"
        elif ma_long > price > ma_short or (ma_short > price and price > ma_long): return "陰天"
        elif ma_short > price and ma_long > price and rsi < config["rsi_bear_threshold"]:
            return "雨天"
        elif ma_short > price and ma_long > price and rsi < config["rsi_oversold"]:
            return "颱風天"
        else: return "陰天"
    except (ValueError, TypeError): return "資料不足"

def get_recovery_status(row, prev_row, config):
    drawdown = row['Drawdown']
    macd_hist = row['MACD_hist']
    prev_drawdown = prev_row['Drawdown'] if prev_row is not None else 0
    try:
        if drawdown > config["drawdown_no_rain"]: return "無雨"
        elif drawdown < config["drawdown_heavy_rain"] and drawdown < prev_drawdown: return "大雨滂沱"
        elif drawdown <= config["drawdown_no_rain"] and drawdown < prev_drawdown: return "陰雨連綿"
        elif drawdown <= config["drawdown_no_rain"] and drawdown > prev_drawdown and macd_hist < 0: return "雨聲漸細"
        elif drawdown <= config["drawdown_no_rain"] and drawdown > prev_drawdown and macd_hist > 0: return "撥雲見日" # 進場訊號
        else: return "無雨"
    except (ValueError, TypeError): return "資料不足"

def run_timing_backtest(df):
    print("\n--- 優化版擇時策略回測 (進場: 撥雲見日, 出場: 雨天/颱風天) ---")
    initial_capital = 100000.0
    capital = initial_capital
    position = 0
    trades = []
    
    buy_price = 0
    buy_date = None

    for i in range(1, len(df)):
        recovery_status = df['Recovery_Status'].iloc[i]
        barometer_status = df['Barometer_Status'].iloc[i]
        price = df['Close'].iloc[i]
        
        if recovery_status == '撥雲見日' and position == 0:
            position = 1
            buy_price = price
            buy_date = df.index[i]
        elif barometer_status in ['雨天', '颱風天'] and position == 1:
            position = 0
            sell_price = price
            sell_date = df.index[i]
            profit = (sell_price - buy_price) / buy_price
            trades.append({'buy_date': buy_date, 'sell_date': sell_date, 'profit': profit})
            capital *= (1 + profit)

    if position == 1:
        sell_price = df['Close'].iloc[-1]
        profit = (sell_price - buy_price) / buy_price
        capital *= (1 + profit)
    
    total_return = (capital - initial_capital) / initial_capital
    print(f"  - 最終資產: ${capital:,.2f}")
    print(f"  - 總報酬率: {total_return:.2%}")
    if trades:
        win_rate = np.mean([1 if t['profit'] > 0 else 0 for t in trades])
        print(f"  - 總交易次數: {len(trades)}")
        print(f"  - 勝率: {win_rate:.2%}")

def run_dca_backtest(df, monthly_investment=10000, signal_multiplier=2.0):
    print(f"\n--- 定期定額加強版策略回測 (每月 ${monthly_investment:,.0f}, 訊號加碼 {signal_multiplier}x) ---")
    total_shares = 0
    total_cash_invested = 0
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    monthly_first_day_df = df.drop_duplicates(subset=['Year', 'Month'], keep='first')
    
    for date, row in monthly_first_day_df.iterrows():
        last_month_date = date - pd.DateOffset(months=1)
        last_month_df = df[(df['Year'] == last_month_date.year) & (df['Month'] == last_month_date.month)]
        
        investment_amount = monthly_investment
        if not last_month_df.empty and '撥雲見日' in last_month_df['Recovery_Status'].values:
            investment_amount *= signal_multiplier

        price = row['Close']
        shares_to_buy = math.floor(investment_amount / price)
        
        if shares_to_buy > 0:
            total_shares += shares_to_buy
            total_cash_invested += shares_to_buy * price
            
    final_portfolio_value = total_shares * df['Close'].iloc[-1]
    total_return = (final_portfolio_value - total_cash_invested) / total_cash_invested if total_cash_invested > 0 else 0

    print(f"  - 總投入現金: ${total_cash_invested:,.2f}")
    print(f"  - 最終資產價值: ${final_portfolio_value:,.2f}")
    print(f"  - 總報酬率: {total_return:.2%}")

def plot_analysis(df, ticker, config_name, config):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.15, 0.15, 0.2])
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='收盤價', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df[f'MA{config["ma_short"]}'], name=f'{config["ma_short"]}日均線', line=dict(color='orange', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df[f'MA{config["ma_long"]}'], name=f'{config["ma_long"]}日均線', line=dict(color='red', dash='dash')), row=1, col=1)

    buy_points = df[df['Recovery_Status'] == '撥雲見日']
    fig.add_trace(go.Scatter(x=buy_points.index, y=buy_points['Close'], mode='markers', marker=dict(color='limegreen', size=10, symbol='triangle-up'), name='撥雲見日 (進場)'), row=1, col=1)

    exit_mask = (df['Barometer_Status'].isin(['雨天', '颱風天'])) & (df['Barometer_Status'].shift(1).isin(['晴天', '多雲', '陰天']))
    exit_points = df[exit_mask]
    fig.add_trace(go.Scatter(x=exit_points.index, y=exit_points['Close'], mode='markers', marker=dict(color='red', size=8, symbol='triangle-down'), name='轉為雨天/颱風天 (出場)'), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=config["rsi_overbought"], line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=config["rsi_oversold"], line_dash="dash", line_color="green", row=2, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='MACD Histogram', marker_color=np.where(df['MACD_hist'] > 0, 'green', 'red')), row=3, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['Drawdown'], name='最大回撤', fill='tozeroy', line=dict(color='grey')), row=4, col=1)
    
    title = f'{ticker} 策略分析 (設定: {config_name})'
    fig.update_layout(title=title, height=800, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    filename = f"stock_strategy_v2_{ticker.replace('.TW', '')}_{config_name}.html"
    fig.write_html(filename)
    print(f"分析完成 ({config_name})，結果已儲存至 {filename}")

if __name__ == '__main__':
    TICKER = "006208.TW"
    base_data = get_stock_data(TICKER)
    
    for name, config in CONFIGS.items():
        print(f"\n{'='*10} 執行分析與回測: 設定組 '{name}' {'='*10}")
        df = base_data.copy()
        df = calculate_indicators(df, config)
        df['Barometer_Status'] = df.apply(lambda row: get_barometer_status(row, config), axis=1)
        
        df['prev_Drawdown'] = df['Drawdown'].shift(1)
        df['Recovery_Status'] = df.apply(lambda row: get_recovery_status(row, {'Drawdown': row['prev_Drawdown']}, config), axis=1)

        print(f"\n--- 基準策略: 買入並持有 (Buy and Hold) ---")
        initial_capital = 100000.0
        buy_and_hold_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
        buy_and_hold_capital = initial_capital * (1 + buy_and_hold_return)
        print(f"  - 最終資產: ${buy_and_hold_capital:,.2f} (以 ${initial_capital:,.0f} 初始資金計算)")
        print(f"  - 總報酬率: {buy_and_hold_return:.2%}")

        run_timing_backtest(df.copy())
        run_dca_backtest(df.copy())
        
        plot_analysis(df, TICKER, name, config)
        print("-" * 50)
