import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(page_title="Stock Portfolio Manager", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# Helper functions for indicators
# ---------------------------
def compute_RSI(data, window=14):
    """Compute Relative Strength Index (RSI)"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_MACD(data, slow=26, fast=12, signal=9):
    """Compute MACD line, signal line, and histogram"""
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_Bollinger_Bands(data, window=20, num_std=2):
    """Compute Bollinger Bands (middle, upper, lower)"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band

def get_stock_data(symbol, period='1y'):
    """Fetch stock data from yfinance"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        if df.empty:
            return None
        return df
    except:
        return None

def get_current_price(symbol):
    """Get latest closing price"""
    try:
        stock = yf.Ticker(symbol)
        # Try to get current price from fast info
        info = stock.info
        price = info.get('currentPrice') or info.get('regularMarketPrice')
        if price is None:
            # Fallback to last close from history
            hist = stock.history(period='1d')
            if not hist.empty:
                price = hist['Close'].iloc[-1]
        return price
    except:
        return None

# ---------------------------
# Initialize session state
# ---------------------------
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=['Symbol', 'Shares', 'Purchase Price'])
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Dashboard", "📈 Portfolio", "👀 Watchlist", "🔬 Analysis"])

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit + yfinance\n\nData delayed by 15-20 min.")

# ---------------------------
# Page: Dashboard
# ---------------------------
if page == "🏠 Dashboard":
    st.title("📊 Portfolio Dashboard")
    
    if st.session_state.portfolio.empty:
        st.warning("Your portfolio is empty. Go to the **Portfolio** page to add stocks.")
    else:
        # Fetch current prices for each holding
        portfolio_data = []
        total_value = 0
        daily_change_total = 0
        failed_symbols = []
        
        for idx, row in st.session_state.portfolio.iterrows():
            symbol = row['Symbol']
            shares = row['Shares']
            purchase_price = row['Purchase Price']
            
            current_price = get_current_price(symbol)
            if current_price is None:
                failed_symbols.append(symbol)
                continue
                
            current_value = shares * current_price
            cost_basis = shares * purchase_price
            gain_loss = current_value - cost_basis
            gain_loss_pct = (gain_loss / cost_basis) * 100 if cost_basis > 0 else 0
            
            # Daily change (approx using last two closes)
            hist = yf.Ticker(symbol).history(period='2d')
            if len(hist) >= 2:
                daily_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100
            else:
                daily_change = 0
            
            portfolio_data.append({
                'Symbol': symbol,
                'Shares': shares,
                'Purchase Price': purchase_price,
                'Current Price': current_price,
                'Current Value': current_value,
                'Gain/Loss ($)': gain_loss,
                'Gain/Loss (%)': gain_loss_pct,
                'Daily Change (%)': daily_change
            })
            total_value += current_value
            daily_change_total += (current_value * daily_change / 100)
        
        # Show warnings for failed fetches
        if failed_symbols:
            st.warning(f"Could not fetch current prices for: {', '.join(failed_symbols)}. They are excluded from calculations.")
        
        # If no valid holdings after filtering, show empty state
        if not portfolio_data:
            st.error("No valid stock data available. Please check your holdings or try again later.")
        else:
            df_portfolio = pd.DataFrame(portfolio_data)
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Portfolio Value", f"${total_value:,.2f}")
            with col2:
                total_gain_loss = df_portfolio['Gain/Loss ($)'].sum()
                total_cost = (df_portfolio['Shares'] * df_portfolio['Purchase Price']).sum()
                total_gain_loss_pct = (total_gain_loss / total_cost) * 100 if total_cost > 0 else 0
                st.metric("Total Gain/Loss", f"${total_gain_loss:,.2f}", delta=f"{total_gain_loss_pct:.2f}%")
            with col3:
                st.metric("Today's Change", f"${daily_change_total:,.2f}", 
                          delta=f"{(daily_change_total/total_value)*100:.2f}%" if total_value > 0 else "0%")
            with col4:
                best_holding = df_portfolio.loc[df_portfolio['Gain/Loss (%)'].idxmax()] if not df_portfolio.empty else None
                worst_holding = df_portfolio.loc[df_portfolio['Gain/Loss (%)'].idxmin()] if not df_portfolio.empty else None
                st.metric("🏆 Best Performer", best_holding['Symbol'] if best_holding is not None else "N/A", 
                          delta=f"{best_holding['Gain/Loss (%)']:.2f}%" if best_holding is not None else None)
            
            # Portfolio holdings table
            st.subheader("📋 Current Holdings")
            st.dataframe(df_portfolio.style.format({
                'Current Value': '${:,.2f}',
                'Gain/Loss ($)': '${:,.2f}',
                'Gain/Loss (%)': '{:.2f}%',
                'Daily Change (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # Allocation pie chart
            if len(df_portfolio) > 0:
                st.subheader("🥧 Portfolio Allocation by Value")
                fig_pie = px.pie(df_portfolio, values='Current Value', names='Symbol', title="")
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Performance over last 30 days
            st.subheader("📈 Performance Over Last 30 Days")
            end = datetime.now()
            start = end - timedelta(days=30)
            combined = pd.DataFrame()
            for symbol in df_portfolio['Symbol'].unique():
                hist = yf.Ticker(symbol).history(start=start, end=end)
                if not hist.empty and len(hist) > 0:
                    norm = hist['Close'] / hist['Close'].iloc[0] * 100
                    combined[symbol] = norm
            if not combined.empty:
                st.line_chart(combined)
            else:
                st.info("Insufficient historical data to display performance chart.")
            
# ---------------------------
# Page: Portfolio Management
# ---------------------------
elif page == "📈 Portfolio":
    st.title("📈 Manage Your Portfolio")
    
    # Add new stock form
    with st.expander("➕ Add New Holding", expanded=True):
        with st.form("add_holding_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                symbol = st.text_input("Stock Symbol", "AAPL").upper()
            with col2:
                shares = st.number_input("Number of Shares", min_value=0.01, step=1.0, value=10.0)
            with col3:
                purchase_price = st.number_input("Purchase Price per Share ($)", min_value=0.01, step=1.0, value=150.0)
            submitted = st.form_submit_button("Add to Portfolio")
            if submitted:
                if symbol and shares > 0 and purchase_price > 0:
                    # Check if symbol already exists
                    existing = st.session_state.portfolio[st.session_state.portfolio['Symbol'] == symbol]
                    if not existing.empty:
                        st.warning(f"{symbol} already in portfolio. Please remove first or edit manually.")
                    else:
                        new_row = pd.DataFrame([[symbol, shares, purchase_price]], 
                                               columns=['Symbol', 'Shares', 'Purchase Price'])
                        st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
                        st.success(f"Added {shares} shares of {symbol} at ${purchase_price:.2f}")
                        st.rerun()
                else:
                    st.error("Please fill all fields correctly.")
    
    # Display and edit current holdings
    st.subheader("✏️ Current Holdings (Editable Table)")
    if not st.session_state.portfolio.empty:
        edited_df = st.data_editor(st.session_state.portfolio, num_rows="dynamic", use_container_width=True)
        if st.button("💾 Save Changes"):
            st.session_state.portfolio = edited_df
            st.success("Portfolio updated!")
    else:
        st.info("Your portfolio is empty. Use the form above to add stocks.")
    
    # Remove stock by symbol
    st.subheader("🗑️ Remove Holding")
    if not st.session_state.portfolio.empty:
        symbol_to_remove = st.selectbox("Select Symbol to Remove", st.session_state.portfolio['Symbol'].unique())
        if st.button("Remove Selected"):
            st.session_state.portfolio = st.session_state.portfolio[st.session_state.portfolio['Symbol'] != symbol_to_remove]
            st.success(f"Removed {symbol_to_remove}")
            st.rerun()
    else:
        st.write("No holdings to remove.")

# ---------------------------
# Page: Watchlist
# ---------------------------
elif page == "👀 Watchlist":
    st.title("👀 Stock Watchlist")
    
    # Add to watchlist
    with st.form("add_watchlist_form"):
        watch_symbol = st.text_input("Add Symbol to Watchlist", "MSFT").upper()
        submitted = st.form_submit_button("➕ Add")
        if submitted:
            if watch_symbol:
                if watch_symbol not in st.session_state.watchlist:
                    # Validate symbol
                    data = get_stock_data(watch_symbol, period='1d')
                    if data is not None and not data.empty:
                        st.session_state.watchlist.append(watch_symbol)
                        st.success(f"Added {watch_symbol} to watchlist.")
                    else:
                        st.error(f"Invalid symbol: {watch_symbol}")
                else:
                    st.warning(f"{watch_symbol} already in watchlist.")
            else:
                st.error("Please enter a symbol.")
    
    # Display watchlist with current prices
    if st.session_state.watchlist:
        st.subheader("📋 Watchlist Items")
        watch_data = []
        for symbol in st.session_state.watchlist:
            price = get_current_price(symbol)
            # Get daily change
            hist = yf.Ticker(symbol).history(period='2d')
            if len(hist) >= 2:
                change_pct = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100
            else:
                change_pct = None
            watch_data.append({
                'Symbol': symbol,
                'Current Price': f"${price:.2f}" if price is not None else "N/A",
                'Daily Change (%)': f"{change_pct:.2f}%" if change_pct is not None else "N/A"
            })
        watch_df = pd.DataFrame(watch_data)
        st.dataframe(watch_df, use_container_width=True)
        
        # Remove from watchlist
        st.subheader("❌ Remove from Watchlist")
        remove_symbol = st.selectbox("Select Symbol to Remove", st.session_state.watchlist)
        if st.button("Remove"):
            st.session_state.watchlist.remove(remove_symbol)
            st.rerun()
    else:
        st.info("Your watchlist is empty. Add some symbols above.")

# ---------------------------
# Page: Analysis (with all indicators)
# ---------------------------
elif page == "🔬 Analysis":
    st.title("🔬 Advanced Stock Analysis")
    st.markdown("Get detailed technical indicators: RSI, MACD, Bollinger Bands, Moving Averages.")
    
    symbol_analysis = st.text_input("Enter Stock Symbol", "AAPL").upper()
    period = st.selectbox("Select Data Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
    
    if symbol_analysis:
        df = get_stock_data(symbol_analysis, period=period)
        if df is None or df.empty:
            st.error(f"No data found for {symbol_analysis}. Please check the symbol.")
        else:
            st.success(f"Showing analysis for {symbol_analysis}")
            
            # Key metrics row
            info = yf.Ticker(symbol_analysis).info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
            with col2:
                pe = info.get('trailingPE', 'N/A')
                st.metric("P/E Ratio", pe if pe == 'N/A' else f"{pe:.2f}")
            with col3:
                market_cap = info.get('marketCap', 'N/A')
                if market_cap != 'N/A':
                    st.metric("Market Cap", f"${market_cap/1e9:.2f}B")
                else:
                    st.metric("Market Cap", "N/A")
            with col4:
                dividend_yield = info.get('dividendYield', 0)
                if dividend_yield:
                    st.metric("Dividend Yield", f"{dividend_yield*100:.2f}%")
                else:
                    st.metric("Dividend Yield", "N/A")
            
            # Price chart with EMAs
            st.subheader("📉 Price Chart with Exponential Moving Averages")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
            # EMA 12 and 26 (common)
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            fig.add_trace(go.Scatter(x=df.index, y=ema12, mode='lines', name='EMA 12', line=dict(color='orange', dash='dash')))
            fig.add_trace(go.Scatter(x=df.index, y=ema26, mode='lines', name='EMA 26', line=dict(color='red', dash='dash')))
            fig.update_layout(title=f"{symbol_analysis} - Price & EMAs", xaxis_title="Date", yaxis_title="Price (USD)", hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
            # RSI
            st.subheader("📊 Relative Strength Index (RSI)")
            rsi = compute_RSI(df['Close'])
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=rsi, mode='lines', name='RSI', line=dict(color='purple')))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
            fig_rsi.update_layout(title=f"RSI (14-day)", xaxis_title="Date", yaxis_title="RSI")
            st.plotly_chart(fig_rsi, use_container_width=True)
            current_rsi = rsi.iloc[-1]
            st.metric("Current RSI", f"{current_rsi:.2f}", 
                      delta="Overbought" if current_rsi > 70 else ("Oversold" if current_rsi < 30 else "Neutral"))
            
            # MACD
            st.subheader("📈 MACD (Moving Average Convergence Divergence)")
            macd_line, signal_line, histogram = compute_MACD(df['Close'])
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df.index, y=macd_line, mode='lines', name='MACD Line', line=dict(color='blue')))
            fig_macd.add_trace(go.Scatter(x=df.index, y=signal_line, mode='lines', name='Signal Line', line=dict(color='red')))
            fig_macd.add_bar(x=df.index, y=histogram, name='Histogram', marker_color='grey')
            fig_macd.update_layout(title=f"MACD (12,26,9)", xaxis_title="Date", yaxis_title="Value")
            st.plotly_chart(fig_macd, use_container_width=True)
            
            # Bollinger Bands
            st.subheader("📉 Bollinger Bands (20-day, 2 std dev)")
            middle_band, upper_band, lower_band = compute_Bollinger_Bands(df['Close'])
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='black')))
            fig_bb.add_trace(go.Scatter(x=df.index, y=upper_band, mode='lines', name='Upper Band', line=dict(color='red', dash='dash')))
            fig_bb.add_trace(go.Scatter(x=df.index, y=middle_band, mode='lines', name='Middle Band (SMA20)', line=dict(color='blue')))
            fig_bb.add_trace(go.Scatter(x=df.index, y=lower_band, mode='lines', name='Lower Band', line=dict(color='green', dash='dash')))
            fig_bb.update_layout(title=f"Bollinger Bands", xaxis_title="Date", yaxis_title="Price (USD)")
            st.plotly_chart(fig_bb, use_container_width=True)
            
            # Volume
            st.subheader("📊 Trading Volume")
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='lightblue'))
            fig_vol.update_layout(title="Volume over time", xaxis_title="Date", yaxis_title="Volume")
            st.plotly_chart(fig_vol, use_container_width=True)
            
            # Additional info: 52-week range, beta, etc.
            st.subheader("ℹ️ Additional Fundamentals")
            col1, col2, col3 = st.columns(3)
            with col1:
                week52low = info.get('fiftyTwoWeekLow', 'N/A')
                week52high = info.get('fiftyTwoWeekHigh', 'N/A')
                st.metric("52-Week Range", f"{week52low} - {week52high}")
            with col2:
                beta = info.get('beta', 'N/A')
                st.metric("Beta", beta if beta == 'N/A' else f"{beta:.2f}")
            with col3:
                avg_volume = info.get('averageVolume', 'N/A')
                st.metric("Avg Volume (3M)", avg_volume if avg_volume == 'N/A' else f"{avg_volume:,}")
