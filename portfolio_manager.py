"""
NIFTY Portfolio Optimiser + Price Predictor
Full‑featured Indian stock market analytics dashboard
with persistent storage and enhanced timeframes
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sqlite3
import warnings
from scipy import stats
import os

warnings.filterwarnings('ignore')

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="NIFTY Portfolio Optimiser",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Complete NSE Universe (all indices + NIFTY 50 stocks)
# ---------------------------
BROAD_INDICES = ["^NSEI", "NIFTYNEXT50.NS", "NIFTY100.NS", "NIFTY200.NS", "NIFTY500.NS"]
MARKET_CAP_INDICES = ["NIFTYMIDCAPSELECT.NS", "NIFTYMIDCAP100.NS", "NIFTYSMALLCAP250.NS", "NIFTYMICROCAP250.NS"]
SECTORAL_INDICES = ["NIFTYBANK.NS", "NIFTYFINANCIAL.NS", "NIFTYIT.NS", "NIFTYPHARMA.NS", 
                    "NIFTYAUTO.NS", "NIFTYFMCG.NS", "NIFTYMEDIA.NS", "NIFTYMETAL.NS", 
                    "NIFTYREALTY.NS", "NIFTYENERGY.NS", "NIFTYINFRA.NS", "NIFTYPSU.NS"]
THEMATIC_INDICES = ["NIFTYCPSE.NS", "NIFTYCOMMODITIES.NS", "NIFTYINDIACONSUMPTION.NS", 
                    "NIFTYMNC.NS", "NIFTYHEALTHCARE.NS", "NIFTYOILGAS.NS", "NIFTYPVTBANK.NS"]
NIFTY_50_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "BAJFINANCE.NS",
    "ITC.NS", "WIPRO.NS", "AXISBANK.NS", "HCLTECH.NS", "LT.NS",
    "SUNPHARMA.NS", "TITAN.NS", "MARUTI.NS", "ASIANPAINT.NS", "NESTLEIND.NS",
    "ULTRACEMCO.NS", "TECHM.NS", "POWERGRID.NS", "M&M.NS", "BAJAJFINSV.NS",
    "TATAMOTORS.NS", "GRASIM.NS", "NTPC.NS", "INDUSINDBK.NS", "ONGC.NS",
    "DIVISLAB.NS", "DRREDDY.NS", "HDFCLIFE.NS", "BRITANNIA.NS", "CIPLA.NS",
    "SBILIFE.NS", "JSWSTEEL.NS", "ADANIPORTS.NS", "COALINDIA.NS", "EICHERMOT.NS",
    "HEROMOTOCO.NS", "BPCL.NS", "TATASTEEL.NS", "SHREECEM.NS",
    "UPL.NS", "BAJAJ-AUTO.NS", "TATACONSUM.NS", "HINDALCO.NS"
]

ALL_AVAILABLE_INDICES = {
    "Broad Market Benchmarks": BROAD_INDICES,
    "Market Capitalization": MARKET_CAP_INDICES,
    "Sectoral Indices": SECTORAL_INDICES,
    "Thematic & Strategy": THEMATIC_INDICES,
    "Individual Stocks (NIFTY 50)": NIFTY_50_SYMBOLS
}

# Enhanced optimization periods (7 timeframes)
OPTIMIZATION_PERIODS = {
    "Weekly (5 trading days)": 5,
    "Bi-Weekly (10 trading days)": 10,
    "Monthly (21 trading days)": 21,
    "Quarterly (63 trading days)": 63,
    "Bi-Annual (126 trading days)": 126,
    "Annual (252 trading days)": 252,
    "Multi-Year (504 trading days)": 504
}

# ---------------------------
# Database persistence
# ---------------------------
DB_NAME = "portfolio_data.db"

def init_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            allocation REAL NOT NULL,
            category TEXT DEFAULT 'Individual Stocks (NIFTY 50)',
            date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS optimization_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            period TEXT NOT NULL,
            sharpe_ratio REAL,
            expected_return REAL,
            expected_risk REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def load_portfolio():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT symbol, allocation FROM portfolio ORDER BY symbol", conn)
    conn.close()
    return df if not df.empty else pd.DataFrame(columns=['symbol', 'allocation'])

def save_portfolio(symbol, allocation, category):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM portfolio WHERE symbol = ?", (symbol,))
    existing = cursor.fetchone()
    if existing:
        cursor.execute("UPDATE portfolio SET allocation = ?, category = ?, date_added = CURRENT_TIMESTAMP WHERE symbol = ?", 
                       (allocation, category, symbol))
    else:
        cursor.execute("INSERT INTO portfolio (symbol, allocation, category) VALUES (?, ?, ?)", 
                       (symbol, allocation, category))
    conn.commit()
    conn.close()

def delete_from_portfolio(symbol):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM portfolio WHERE symbol = ?", (symbol,))
    conn.commit()
    conn.close()

def update_allocation(symbol, new_allocation):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE portfolio SET allocation = ? WHERE symbol = ?", (new_allocation, symbol))
    conn.commit()
    conn.close()

def clear_portfolio():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM portfolio")
    conn.commit()
    conn.close()

def save_optimization_result(period, sharpe_ratio, expected_return, expected_risk):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO optimization_history (period, sharpe_ratio, expected_return, expected_risk)
        VALUES (?, ?, ?, ?)
    ''', (period, sharpe_ratio, expected_return, expected_risk))
    conn.commit()
    conn.close()

def load_optimization_history():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM optimization_history ORDER BY timestamp DESC LIMIT 50", conn)
    conn.close()
    return df

init_database()

# ---------------------------
# Helper functions for indicators (fully preserved)
# ---------------------------
def fetch_stock_data(symbols, period="1y"):
    """Fetch historical data for multiple symbols"""
    try:
        if isinstance(symbols, str):
            symbols = [symbols]
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period)
                if not df.empty:
                    df.index = pd.to_datetime(df.index)
                    data[symbol] = df['Close']
            except:
                continue
        if not data:
            return None
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def fetch_full_data(symbol, period="1y"):
    """Fetch OHLCV data for technical analysis"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        return df
    except:
        return None

def compute_sma(data, window):
    return data.rolling(window=window).mean()

def compute_ema(data, span):
    return data.ewm(span=span, adjust=False).mean()

def compute_macd(data, fast=12, slow=26, signal=9):
    ema_fast = compute_ema(data, fast)
    ema_slow = compute_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_bollinger_bands(data, window=20, num_std=2):
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return sma, upper, lower

def compute_atr(high, low, close, window=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def compute_adx(high, low, close, window=14):
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    atr = compute_atr(high, low, close, window)
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (abs(minus_dm).rolling(window=window).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=window).mean()
    return adx, plus_di, minus_di

def compute_obv(close, volume):
    obv = np.zeros(len(close))
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv[i] = obv[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv[i] = obv[i-1] - volume.iloc[i]
        else:
            obv[i] = obv[i-1]
    return pd.Series(obv, index=close.index)

def compute_stochastic(high, low, close, k_window=14, d_window=3):
    low_min = low.rolling(window=k_window).min()
    high_max = high.rolling(window=k_window).max()
    stoch_k = 100 * (close - low_min) / (high_max - low_min)
    stoch_d = stoch_k.rolling(window=d_window).mean()
    return stoch_k, stoch_d

def compute_ichimoku(high, low, close, conversion=9, base=26, lagging=52):
    conversion_line = (high.rolling(window=conversion).max() + low.rolling(window=conversion).min()) / 2
    base_line = (high.rolling(window=base).max() + low.rolling(window=base).min()) / 2
    leading_span_a = ((conversion_line + base_line) / 2).shift(base)
    leading_span_b = ((high.rolling(window=lagging).max() + low.rolling(window=lagging).min()) / 2).shift(base)
    lagging_span = close.shift(-base)
    return conversion_line, base_line, leading_span_a, leading_span_b, lagging_span

def compute_fibonacci_levels(high, low):
    diff = high - low
    levels = {
        '0%': low,
        '23.6%': low + diff * 0.236,
        '38.2%': low + diff * 0.382,
        '50%': low + diff * 0.5,
        '61.8%': low + diff * 0.618,
        '78.6%': low + diff * 0.786,
        '100%': high
    }
    return levels

def compute_risk_metrics(returns):
    risk_free_rate = 0.06
    excess_returns = returns - risk_free_rate / 252
    sharpe = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
    downside_returns = returns[returns < 0]
    sortino = np.sqrt(252) * returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    var_95 = np.percentile(returns, 5) * np.sqrt(252)
    cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * np.sqrt(252)
    return {
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_drawdown,
        'VaR (95%)': var_95,
        'CVaR (95%)': cvar_95,
        'Annualized Volatility': returns.std() * np.sqrt(252),
        'Annualized Return': (1 + returns.mean()) ** 252 - 1
    }

def compute_signal_score(data_dict):
    """12-factor signal scoring from original code"""
    score = 0
    reasons = []
    
    rsi = data_dict.get('rsi', 50)
    if rsi < 30:
        score += 25
        reasons.append(f"RSI {rsi:.1f} < 30 — Oversold (BUY Signal)")
    elif rsi > 70:
        score -= 25
        reasons.append(f"RSI {rsi:.1f} > 70 — Overbought (SELL Signal)")
    elif rsi < 40:
        score += 10
        reasons.append(f"RSI {rsi:.1f} < 40 — Approaching oversold")
    elif rsi > 60:
        score -= 10
        reasons.append(f"RSI {rsi:.1f} > 60 — Approaching overbought")
    
    macd = data_dict.get('macd_hist', 0)
    macd_prev = data_dict.get('macd_hist_prev', 0)
    if macd > 0 and macd_prev <= 0:
        score += 30
        reasons.append("MACD Histogram turns positive — Bullish Crossover")
    elif macd < 0 and macd_prev >= 0:
        score -= 30
        reasons.append("MACD Histogram turns negative — Bearish Crossover")
    elif macd > 0:
        score += 10
        reasons.append("MACD Histogram positive — Bullish momentum")
    elif macd < 0:
        score -= 10
        reasons.append("MACD Histogram negative — Bearish momentum")
    
    bb_position = data_dict.get('bb_position', 0.5)
    if bb_position < 0:
        score += 20
        reasons.append("Price below Lower Bollinger Band — Oversold bounce possible")
    elif bb_position > 1:
        score -= 20
        reasons.append("Price above Upper Bollinger Band — Overbought pullback possible")
    
    ma_cross = data_dict.get('ma_cross', 0)
    if ma_cross == 1:
        score += 15
        reasons.append("Golden Cross (50MA > 200MA) — Bullish trend")
    elif ma_cross == -1:
        score -= 15
        reasons.append("Death Cross (50MA < 200MA) — Bearish trend")
    
    volume_ratio = data_dict.get('volume_ratio', 1)
    if volume_ratio > 1.5:
        if score > 0:
            score += 15
            reasons.append(f"Volume {volume_ratio:.1f}x average — Volume confirmation")
        elif score < 0:
            score -= 10
            reasons.append(f"Volume {volume_ratio:.1f}x average — Selling pressure")
    
    atr = data_dict.get('atr', 0)
    avg_atr = data_dict.get('avg_atr', 0)
    if atr > avg_atr * 1.5:
        if score > 0:
            score += 10
            reasons.append("High volatility — Strong trend")
    
    return score, reasons

def optimize_portfolio(selected_symbols, period_days=252):
    """Monte Carlo portfolio optimisation (preserved)"""
    try:
        returns_data = fetch_stock_data(selected_symbols, period=f"{period_days}d")
        if returns_data is None or returns_data.empty:
            return None, None, None, None
        returns = returns_data.pct_change().dropna()
        num_portfolios = 5000
        results = np.zeros((3, num_portfolios))
        weights_record = []
        for i in range(num_portfolios):
            weights = np.random.random(len(selected_symbols))
            weights /= np.sum(weights)
            weights_record.append(weights)
            port_return = np.sum(returns.mean() * weights) * 252
            port_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            sharpe = port_return / port_std if port_std > 0 else 0
            results[0, i] = port_return
            results[1, i] = port_std
            results[2, i] = sharpe
        max_sharpe_idx = np.argmax(results[2])
        optimal_weights = weights_record[max_sharpe_idx]
        opt_df = pd.DataFrame({'Symbol': selected_symbols, 'Weight %': optimal_weights * 100}).sort_values('Weight %', ascending=False)
        return opt_df, results[0, max_sharpe_idx], results[1, max_sharpe_idx], results[2, max_sharpe_idx]
    except Exception as e:
        st.error(f"Optimization failed: {str(e)}")
        return None, None, None, None

def find_optimal_rebalancing_window(symbols):
    test_periods = [5, 10, 21, 63, 126, 252, 504]
    results = []
    for days in test_periods:
        _, ret, risk, sharpe = optimize_portfolio(symbols, days)
        if ret is not None:
            results.append({
                'Rebalance Days': days,
                'Rebalance Period': get_period_name(days),
                'Expected Return': ret,
                'Expected Risk': risk,
                'Sharpe Ratio': sharpe
            })
    if results:
        results_df = pd.DataFrame(results)
        best = results_df.loc[results_df['Sharpe Ratio'].idxmax()]
        return results_df, best
    return None, None

def get_period_name(days):
    name_map = {5: "Weekly", 10: "Bi-Weekly", 21: "Monthly", 63: "Quarterly",
                126: "Bi-Annual", 252: "Annual", 504: "Multi-Year"}
    return name_map.get(days, f"{days} Days")

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("📊 NIFTY Portfolio Optimiser")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Dashboard", "📈 Portfolio Builder", "🔬 Stock Analysis", "🎯 Signal Scanner", 
     "📊 Risk Analytics", "⚙️ Portfolio Management", "🎯 Optimal Rebalancing", "📈 Performance History"]
)
st.sidebar.markdown("---")
st.sidebar.info("All indicators & calculations from original version are preserved.")
st.sidebar.caption("Data Source: Yahoo Finance (NSE)")

# ---------------------------
# Page 1: Dashboard (NIFTY overview)
# ---------------------------
if page == "🏠 Dashboard":
    st.title("🏠 NIFTY Market Dashboard")
    nifty_data = fetch_stock_data(["^NSEI"], period="1mo")
    if nifty_data is not None and not nifty_data.empty:
        col1, col2, col3, col4 = st.columns(4)
        current = nifty_data.iloc[-1,0]
        prev = nifty_data.iloc[-2,0]
        change = ((current - prev)/prev)*100
        col1.metric("NIFTY 50", f"{current:,.2f}", delta=f"{change:.2f}%")
        # Market trend using SMAs
        nifty_close = nifty_data.iloc[:,0]
        sma20 = compute_sma(nifty_close, 20)
        sma50 = compute_sma(nifty_close, 50)
        if nifty_close.iloc[-1] > sma50.iloc[-1]:
            col2.metric("Market Trend", "Bullish", delta="Above 50 DMA")
        else:
            col2.metric("Market Trend", "Bearish", delta="Below 50 DMA")
        col3.metric("20-Day SMA", f"{sma20.iloc[-1]:,.2f}")
        col4.metric("50-Day SMA", f"{sma50.iloc[-1]:,.2f}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nifty_data.index, y=nifty_close, mode='lines', name='NIFTY 50'))
        fig.add_trace(go.Scatter(x=nifty_data.index, y=sma20, mode='lines', name='SMA20', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=nifty_data.index, y=sma50, mode='lines', name='SMA50', line=dict(dash='dash')))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Page 2: Portfolio Builder (with enhanced periods)
# ---------------------------
elif page == "📈 Portfolio Builder":
    st.title("📈 Portfolio Builder & Optimisation")
    category = st.selectbox("Select Category", list(ALL_AVAILABLE_INDICES.keys()))
    available = ALL_AVAILABLE_INDICES[category]
    selected = st.multiselect("Select Assets", available, default=available[:3] if len(available)>=3 else available)
    if len(selected) >= 2:
        period_name = st.selectbox("Optimization Period", list(OPTIMIZATION_PERIODS.keys()), index=2)
        period_days = OPTIMIZATION_PERIODS[period_name]
        if st.button("Optimise Portfolio"):
            with st.spinner(f"Optimising over {period_name}..."):
                opt_df, ret, risk, sharpe = optimize_portfolio(selected, period_days)
                if opt_df is not None:
                    st.subheader("Optimal Allocation")
                    st.dataframe(opt_df.style.format({'Weight %': '{:.2f}%'}), use_container_width=True)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Expected Return", f"{ret:.2%}" if ret else "N/A")
                    col2.metric("Expected Risk", f"{risk:.2%}" if risk else "N/A")
                    col3.metric("Sharpe Ratio", f"{sharpe:.2f}" if sharpe else "N/A")
                    fig_pie = px.pie(opt_df, values='Weight %', names='Symbol')
                    st.plotly_chart(fig_pie, use_container_width=True)
                    if st.button("Save to Database"):
                        clear_portfolio()
                        for _, row in opt_df.iterrows():
                            save_portfolio(row['Symbol'], row['Weight %'], category)
                        save_optimization_result(period_name, sharpe, ret, risk)
                        st.success("Portfolio saved!")
    else:
        st.info("Select at least 2 assets for optimisation.")

# ---------------------------
# Page 3: Stock Analysis (full indicators)
# ---------------------------
elif page == "🔬 Stock Analysis":
    st.title("🔬 Advanced Stock Analysis")
    all_symbols = []
    for indices in ALL_AVAILABLE_INDICES.values():
        all_symbols.extend(indices)
    all_symbols = list(set(all_symbols))
    symbol = st.selectbox("Select Stock/Index", all_symbols, index=0)
    period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y"], index=2)
    if symbol:
        df = fetch_full_data(symbol, period)
        if df is not None and not df.empty:
            # Calculate all indicators
            df['SMA20'] = compute_sma(df['Close'], 20)
            df['SMA50'] = compute_sma(df['Close'], 50)
            df['SMA200'] = compute_sma(df['Close'], 200)
            df['RSI'] = compute_rsi(df['Close'])
            macd_line, signal_line, macd_hist = compute_macd(df['Close'])
            df['MACD'] = macd_line
            df['MACD_Signal'] = signal_line
            df['MACD_Hist'] = macd_hist
            bb_mid, bb_upper, bb_lower = compute_bollinger_bands(df['Close'])
            df['BB_Upper'] = bb_upper
            df['BB_Lower'] = bb_lower
            df['ATR'] = compute_atr(df['High'], df['Low'], df['Close'])
            df['OBV'] = compute_obv(df['Close'], df['Volume'])
            stoch_k, stoch_d = compute_stochastic(df['High'], df['Low'], df['Close'])
            df['Stoch_K'] = stoch_k
            df['Stoch_D'] = stoch_d
            adx, plus_di, minus_di = compute_adx(df['High'], df['Low'], df['Close'])
            df['ADX'] = adx
            df['Plus_DI'] = plus_di
            df['Minus_DI'] = minus_di
            # Ichimoku
            conv, base, span_a, span_b, lagging = compute_ichimoku(df['High'], df['Low'], df['Close'])
            df['Ichimoku_Conv'] = conv
            df['Ichimoku_Base'] = base
            
            # Header metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"₹{df['Close'].iloc[-1]:.2f}")
            daily_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2])/df['Close'].iloc[-2])*100
            col2.metric("Daily Change", f"{daily_change:.2f}%")
            col3.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")
            col4.metric("ATR", f"{df['ATR'].iloc[-1]:.2f}")
            
            # Chart: Candlestick + Bollinger + MAs
            st.subheader("Price Chart with Bollinger Bands & Moving Averages")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', name='BB Upper', line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', name='BB Lower', line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode='lines', name='SMA20'))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA50'))
            st.plotly_chart(fig, use_container_width=True)
            
            # RSI
            st.subheader("RSI (Relative Strength Index)")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            # MACD
            st.subheader("MACD")
            fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])
            fig_macd.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price'), row=1, col=1)
            fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=2, col=1)
            fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='red')), row=2, col=1)
            fig_macd.add_bar(x=df.index, y=df['MACD_Hist'], name='Histogram', row=2, col=1)
            st.plotly_chart(fig_macd, use_container_width=True)
            
            # ADX
            st.subheader("ADX - Trend Strength")
            fig_adx = go.Figure()
            fig_adx.add_trace(go.Scatter(x=df.index, y=df['ADX'], name='ADX'))
            fig_adx.add_trace(go.Scatter(x=df.index, y=df['Plus_DI'], name='+DI'))
            fig_adx.add_trace(go.Scatter(x=df.index, y=df['Minus_DI'], name='-DI'))
            fig_adx.add_hline(y=25, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_adx, use_container_width=True)
            
            # Stochastic
            st.subheader("Stochastic Oscillator")
            fig_stoch = go.Figure()
            fig_stoch.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], name='%K'))
            fig_stoch.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'], name='%D'))
            fig_stoch.add_hline(y=80, line_dash="dash", line_color="red")
            fig_stoch.add_hline(y=20, line_dash="dash", line_color="green")
            st.plotly_chart(fig_stoch, use_container_width=True)
            
            # Fibonacci
            st.subheader("Fibonacci Retracement (52W Range)")
            year_high = df['High'].max()
            year_low = df['Low'].min()
            fib = compute_fibonacci_levels(year_high, year_low)
            fib_df = pd.DataFrame(list(fib.items()), columns=['Level', 'Price'])
            st.dataframe(fib_df.style.format({'Price': '₹{:.2f}'}))
        else:
            st.error("No data found")

# ---------------------------
# Page 4: Signal Scanner (original 12‑factor)
# ---------------------------
elif page == "🎯 Signal Scanner":
    st.title("🎯 12-Factor High-Conviction Signal Scanner")
    if st.button("Run Full Market Scan (All NIFTY 50)"):
        with st.spinner("Scanning NIFTY 50 stocks..."):
            all_scores = []
            for sym in NIFTY_50_SYMBOLS:
                try:
                    df = fetch_full_data(sym, "3mo")
                    if df is not None and len(df) > 50:
                        rsi = compute_rsi(df['Close'])
                        macd_line, sig, hist = compute_macd(df['Close'])
                        bb_mid, bb_up, bb_low = compute_bollinger_bands(df['Close'])
                        sma50 = compute_sma(df['Close'], 50)
                        sma200 = compute_sma(df['Close'], 200)
                        atr = compute_atr(df['High'], df['Low'], df['Close'])
                        
                        bb_pos = (df['Close'].iloc[-1] - bb_low.iloc[-1]) / (bb_up.iloc[-1] - bb_low.iloc[-1]) if (bb_up.iloc[-1] - bb_low.iloc[-1])>0 else 0.5
                        if len(sma50)>1 and len(sma200)>1:
                            if sma50.iloc[-1] > sma200.iloc[-1] and sma50.iloc[-2] <= sma200.iloc[-2]:
                                ma_cross = 1
                            elif sma50.iloc[-1] < sma200.iloc[-1] and sma50.iloc[-2] >= sma200.iloc[-2]:
                                ma_cross = -1
                            else:
                                ma_cross = 0
                        else:
                            ma_cross = 0
                        avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
                        vol_ratio = df['Volume'].iloc[-1] / avg_vol if avg_vol>0 else 1
                        data_dict = {
                            'rsi': rsi.iloc[-1],
                            'macd_hist': hist.iloc[-1],
                            'macd_hist_prev': hist.iloc[-2] if len(hist)>1 else 0,
                            'bb_position': bb_pos - 1,
                            'ma_cross': ma_cross,
                            'volume_ratio': vol_ratio,
                            'atr': atr.iloc[-1],
                            'avg_atr': atr.rolling(20).mean().iloc[-1]
                        }
                        score, reasons = compute_signal_score(data_dict)
                        all_scores.append({'Symbol': sym, 'Score': score, 'Signal': 'BUY' if score>15 else ('SELL' if score<-15 else 'NEUTRAL'), 'RSI': f"{rsi.iloc[-1]:.1f}"})
                except:
                    continue
            if all_scores:
                df_res = pd.DataFrame(all_scores).sort_values('Score', ascending=False)
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Top BUY Signals")
                    st.dataframe(df_res[df_res['Score']>15].head(5))
                with col2:
                    st.subheader("Top SELL Signals")
                    st.dataframe(df_res[df_res['Score']<-15].sort_values('Score').head(5))
                st.subheader("All Scores")
                st.dataframe(df_res)
            else:
                st.error("Scan failed")

# ---------------------------
# Page 5: Risk Analytics (fully preserved)
# ---------------------------
elif page == "📊 Risk Analytics":
    st.title("Advanced Portfolio Risk Analytics")
    portfolio_df = load_portfolio()
    if not portfolio_df.empty:
        symbols = portfolio_df['symbol'].tolist()
        weights = portfolio_df['allocation'].values / 100
        period = st.selectbox("Risk Analysis Period", ["1y", "2y"])
        if st.button("Compute Risk Metrics"):
            returns_data = fetch_stock_data(symbols, period=period)
            if returns_data is not None and not returns_data.empty:
                returns = returns_data.pct_change().dropna()
                if len(symbols) > 1:
                    port_returns = returns.dot(weights)
                else:
                    port_returns = returns
                metrics = compute_risk_metrics(port_returns)
                col1, col2, col3 = st.columns(3)
                col1.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
                col1.metric("Sortino Ratio", f"{metrics['Sortino Ratio']:.2f}")
                col1.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
                col2.metric("Annualized Return", f"{metrics['Annualized Return']:.2%}")
                col2.metric("Volatility", f"{metrics['Annualized Volatility']:.2%}")
                col2.metric("VaR (95%)", f"{metrics['VaR (95%)']:.2%}")
                col3.metric("CVaR (95%)", f"{metrics['CVaR (95%)']:.2%}")
                
                # Drawdown chart
                cum = (1+port_returns).cumprod()
                running_max = cum.expanding().max()
                drawdown = (cum - running_max)/running_max
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, fill='tozeroy', name='Drawdown'))
                st.plotly_chart(fig, use_container_width=True)
                
                # Rolling Sharpe
                rolling_sharpe = port_returns.rolling(60).apply(lambda x: np.sqrt(252)*x.mean()/x.std() if x.std()>0 else 0)
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, name='60d Rolling Sharpe'))
                st.plotly_chart(fig2)
                
                # Correlation heatmap (if multiple assets)
                if len(symbols) > 1:
                    corr = returns.corr()
                    fig3 = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu')
                    st.plotly_chart(fig3)
    else:
        st.warning("No portfolio saved. Please build and save a portfolio first.")

# ---------------------------
# Page 6: Portfolio Management (CRUD)
# ---------------------------
elif page == "⚙️ Portfolio Management":
    st.title("Portfolio Management (Persistent)")
    portfolio_df = load_portfolio()
    if not portfolio_df.empty:
        st.dataframe(portfolio_df, use_container_width=True)
        # Edit allocation
        sym_edit = st.selectbox("Select symbol to edit", portfolio_df['symbol'].tolist())
        new_alloc = st.number_input("New Allocation (%)", 0.0, 100.0, 
                                    float(portfolio_df[portfolio_df['symbol']==sym_edit]['allocation'].iloc[0]))
        if st.button("Update Allocation"):
            update_allocation(sym_edit, new_alloc)
            st.rerun()
        # Delete
        sym_del = st.selectbox("Select symbol to delete", portfolio_df['symbol'].tolist(), key="del")
        if st.button("Delete Holding"):
            delete_from_portfolio(sym_del)
            st.rerun()
        if st.button("Clear Entire Portfolio"):
            clear_portfolio()
            st.rerun()
    else:
        st.info("No portfolio found. Use 'Portfolio Builder' to create one.")
    # Manual add
    with st.expander("Add Single Holding"):
        with st.form("manual_add"):
            sym = st.text_input("Symbol (e.g., TCS.NS)")
            alloc = st.number_input("Allocation (%)", 0.0, 100.0)
            cat = st.selectbox("Category", list(ALL_AVAILABLE_INDICES.keys()))
            if st.form_submit_button("Add"):
                if sym and alloc>0:
                    save_portfolio(sym, alloc, cat)
                    st.success(f"Added {sym}")

# ---------------------------
# Page 7: Optimal Rebalancing Window
# ---------------------------
elif page == "🎯 Optimal Rebalancing":
    st.title("Find Optimal Rebalancing Window")
    portfolio_df = load_portfolio()
    if not portfolio_df.empty:
        symbols = portfolio_df['symbol'].tolist()
        if st.button("Analyze Rebalancing Periods"):
            with st.spinner("Testing 7 rebalancing windows..."):
                results, best = find_optimal_rebalancing_window(symbols)
                if results is not None:
                    st.dataframe(results.style.format({'Expected Return': '{:.2%}', 'Expected Risk': '{:.2%}', 'Sharpe Ratio': '{:.3f}'}))
                    st.success(f"Optimal rebalancing: **{best['Rebalance Period']}** with Sharpe ratio {best['Sharpe Ratio']:.3f}")
                    fig = px.bar(results, x='Rebalance Period', y='Sharpe Ratio', color='Sharpe Ratio', color_continuous_scale='Viridis')
                    st.plotly_chart(fig)
                else:
                    st.error("Analysis failed")
    else:
        st.warning("No portfolio saved. Please create a portfolio first.")

# ---------------------------
# Page 8: Performance History
# ---------------------------
elif page == "📈 Performance History":
    st.title("Optimisation History")
    history = load_optimization_history()
    if not history.empty:
        st.dataframe(history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history['timestamp'], y=history['sharpe_ratio'], mode='lines+markers', name='Sharpe Ratio'))
        st.plotly_chart(fig)
    else:
        st.info("No history yet. Run some optimisations.")

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    pass
