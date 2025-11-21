import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ephem
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# --- CONFIGURATION ---
st.set_page_config(page_title="Astro-Quant Ingress Scanner", layout="wide")

# --- 1. ASTROLOGY ENGINE (INGRESS FOCUSED) ---
def get_astro_data(target_date):
    """
    Calculates:
    1. New Aspects (Started today)
    2. Ingress Events (Planet entered a new sign today)
    3. Retrograde Status (Which planets are currently Rx)
    """
    d = target_date.to_pydatetime()
    
    planet_names = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 
                   'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']
    
    zodiacs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 
               'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']

    # --- Helper to capture state ---
    def get_state(check_date):
        obs = ephem.Observer()
        obs.date = check_date
        state = {}
        for name in planet_names:
            body = getattr(ephem, name)()
            body.compute(obs)
            lon = np.degrees(ephem.Ecliptic(body).lon)
            state[name] = {'lon': lon, 'sign_idx': int(lon / 30)}
        return state

    # 1. Get States (Today vs Yesterday)
    current_state = get_state(d)
    prev_state = get_state(d - timedelta(days=1))
    
    ingress_list = []
    retro_list = []
    
    # 2. Analyze Planets
    for name in planet_names:
        # A. INGRESS CHECK
        curr_sign_idx = current_state[name]['sign_idx']
        prev_sign_idx = prev_state[name]['sign_idx']
        
        if curr_sign_idx != prev_sign_idx:
            new_sign = zodiacs[curr_sign_idx % 12]
            ingress_list.append(f"{name} enters {new_sign}")
            
        # B. RETROGRADE CHECK
        obs_now = ephem.Observer(); obs_now.date = d
        obs_next = ephem.Observer(); obs_next.date = d + timedelta(hours=1)
        body_now = getattr(ephem, name)(); body_now.compute(obs_now)
        body_next = getattr(ephem, name)(); body_next.compute(obs_next)
        
        if body_next.g_ra < body_now.g_ra:
            retro_list.append(f"{name} Rx")

    # 3. NEW ASPECTS CHECK
    def get_aspects(state_dict):
        found = set()
        names = list(state_dict.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                p1, p2 = names[i], names[j]
                l1, l2 = state_dict[p1]['lon'], state_dict[p2]['lon']
                diff = abs(l1 - l2)
                if diff > 180: diff = 360 - diff
                if abs(diff - 0) < 3.0: found.add(f"{p1} conj {p2}")
                elif abs(diff - 180) < 3.0: found.add(f"{p1} opp {p2}")
                elif abs(diff - 120) < 3.0: found.add(f"{p1} tri {p2}")
                elif abs(diff - 90) < 3.0: found.add(f"{p1} sq {p2}")
        return found

    asp_curr = get_aspects(current_state)
    asp_prev = get_aspects(prev_state)
    new_aspects = list(asp_curr - asp_prev)

    return {
        "New_Aspects": ", ".join(new_aspects),
        "Ingress": ", ".join(ingress_list),
        "Retrograde": ", ".join(retro_list)
    }

# --- 2. ZIGZAG ALGORITHM ---
def calculate_zigzag(df, deviation_percent):
    tmp_max = float(df.iloc[0]['High'])
    tmp_min = float(df.iloc[0]['Low'])
    tmp_max_idx = df.index[0]
    tmp_min_idx = df.index[0]
    
    trend = 0 
    pivots = [] 
    multiplier = deviation_percent / 100.0

    for date, row in df.iterrows():
        current_high = float(row['High'])
        current_low = float(row['Low'])
        
        if trend == 0:
            if current_high >= tmp_min * (1 + multiplier):
                trend = 1
                tmp_max = current_high
                tmp_max_idx = date
                pivots.append({'Date': tmp_min_idx, 'Price': tmp_min, 'Type': 'Low'})
            elif current_low <= tmp_max * (1 - multiplier):
                trend = -1
                tmp_min = current_low
                tmp_min_idx = date
                pivots.append({'Date': tmp_max_idx, 'Price': tmp_max, 'Type': 'High'})
        
        elif trend == 1:
            if current_high > tmp_max:
                tmp_max = current_high
                tmp_max_idx = date
            if current_low < tmp_max * (1 - multiplier):
                trend = -1
                pivots.append({'Date': tmp_max_idx, 'Price': tmp_max, 'Type': 'High'})
                tmp_min = current_low
                tmp_min_idx = date
                
        elif trend == -1:
            if current_low < tmp_min:
                tmp_min = current_low
                tmp_min_idx = date
            if current_high > tmp_min * (1 + multiplier):
                trend = 1
                pivots.append({'Date': tmp_min_idx, 'Price': tmp_min, 'Type': 'Low'})
                tmp_max = current_high
                tmp_max_idx = date

    if trend == 1:
        pivots.append({'Date': tmp_max_idx, 'Price': tmp_max, 'Type': 'High'})
    elif trend == -1:
        pivots.append({'Date': tmp_min_idx, 'Price': tmp_min, 'Type': 'Low'})
        
    return pd.DataFrame(pivots)

# --- 3. ANALYTICAL FUNCTIONS ---
def analyze_correlations(df):
    stats = {
        'Aspects': [],
        'Ingress': [],
        'Retrograde': []
    }
    
    pivots_with_trigger = 0
    total_pivots = len(df)
    
    for index, row in df.iterrows():
        p_type = row['Type']
        triggered = False
        
        # Aspects
        if row['New Aspects']:
            triggered = True
            for item in row['New Aspects'].split(', '):
                stats['Aspects'].append({'Event': item, 'Type': p_type})
        
        # Ingress
        if row['Ingress Events']:
            triggered = True
            for item in row['Ingress Events'].split(', '):
                stats['Ingress'].append({'Event': item, 'Type': p_type})

        # Retrograde
        if row['Retrograde Status']:
            for item in row['Retrograde Status'].split(', '):
                stats['Retrograde'].append({'Event': item, 'Type': p_type})
        
        if triggered:
            pivots_with_trigger += 1
            
    return stats, pivots_with_trigger, total_pivots

# --- 4. UI & MAIN LOGIC ---
st.title("📊 Astro-Quant Ingress Scanner")
st.markdown("Analyze Market Reversals based on **Ingress Events** (Sign Changes) and **New Aspects**.")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Stock Ticker (Yahoo)", value="^NSEI")
    period_map = {"1 Year": "1y", "2 Years": "2y", "5 Years": "5y", "Max": "max"}
    period_sel = st.selectbox("Date Range", options=list(period_map.keys()))
    deviation = st.number_input("ZigZag Deviation (%)", min_value=1.0, value=5.0, step=0.5)
    
    if st.button("Analyze Stock"):
        with st.spinner('Calculating Pivots & Scanning Ingresses...'):
            try:
                df = yf.download(ticker, period=period_map[period_sel], progress=False)
                
                if df.empty:
                    st.error("No data found for this ticker.")
                else:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    
                    pivot_df = calculate_zigzag(df, deviation)
                    
                    if pivot_df.empty:
                        st.warning("No pivots found. Try lowering deviation %.")
                        if 'final_df' in st.session_state: del st.session_state['final_df']
                    else:
                        astro_data_list = []
                        prog_bar = st.progress(0)
                        total_pivots = len(pivot_df)
                        
                        for i, row in pivot_df.iterrows():
                            p_date = row['Date']
                            astro = get_astro_data(p_date)
                            
                            flat_astro = {
                                "Date": p_date.date(),
                                "Type": row['Type'],
                                "Price": round(row['Price'], 2),
                                "New Aspects": astro['New_Aspects'],
                                "Ingress Events": astro['Ingress'],
                                "Retrograde Status": astro['Retrograde']
                            }
                            astro_data_list.append(flat_astro)
                            prog_bar.progress((i + 1) / total_pivots)

                        final_df = pd.DataFrame(astro_data_list)
                        final_df = final_df.sort_values(by="Date", ascending=False)
                        
                        stats_data, p_with, p_total = analyze_correlations(final_df)
                        
                        st.session_state['final_df'] = final_df
                        st.session_state['pivot_df'] = pivot_df
                        st.session_state['stock_df'] = df
                        st.session_state['stats_data'] = stats_data
                        st.session_state['p_with'] = p_with
                        st.session_state['p_total'] = p_total
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- 5. RENDER RESULTS ---
if 'final_df' in st.session_state:
    final_df = st.session_state['final_df']
    pivot_df = st.session_state['pivot_df']
    df = st.session_state['stock_df']
    stats_data = st.session_state['stats_data']
    p_with = st.session_state['p_with']
    p_total = st.session_state['p_total']

    tab1, tab2 = st.tabs(["📉 Visuals & Data Table", "🧠 Ingress Insights"])
    
    with tab1:
        # Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index,
                        open=df['Open'], high=df['High'],
                        low=df['Low'], close=df['Close'],
                        name='Price', opacity=0.4))
        
        fig.add_trace(go.Scatter(x=pivot_df['Date'], 
                                 y=pivot_df['Price'], 
                                 mode='lines',
                                 name=f'ZigZag ({deviation}%)',
                                 line=dict(color='blue', width=1)))
        
        highs = pivot_df[pivot_df['Type']=='High']
        lows = pivot_df[pivot_df['Type']=='Low']
        fig.add_trace(go.Scatter(x=highs['Date'], y=highs['Price'], mode='markers', name='Tops', marker=dict(color='red', size=8, symbol='triangle-down')))
        fig.add_trace(go.Scatter(x=lows['Date'], y=lows['Price'], mode='markers', name='Bottoms', marker=dict(color='green', size=8, symbol='triangle-up')))

        fig.update_layout(height=600, xaxis_rangeslider_visible=False, title=f"{ticker} Pivots")
        st.plotly_chart(fig, use_container_width=True)
        
        # The Clean Table
        st.subheader("Raw Data: Pivots & Ingress Triggers")
        st.dataframe(final_df, use_container_width=True)
        
        csv = final_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data as CSV", csv, "astro_ingress_data.csv", "text/csv")

    with tab2:
        st.header("Deep Analysis: Do Ingresses Move the Market?")
        
        eff_ratio = (p_with / p_total) * 100 if p_total > 0 else 0
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total ZigZag Pivots", p_total)
        m2.metric("Pivots on Ingress/Aspect Day", p_with)
        m3.metric("Astro Correlation Efficiency", f"{eff_ratio:.1f}%")
        
        st.markdown("---")

        # --- 1. INGRESS ANALYSIS ---
        st.subheader("Top Ingress Triggers (Sign Entries)")
        if stats_data['Ingress']:
            df_ing = pd.DataFrame(stats_data['Ingress'])
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("### 🟢 Top BUY Ingresses")
                buy_ing = df_ing[df_ing['Type'] == 'Low']
                if not buy_ing.empty:
                    counts = buy_ing['Event'].value_counts().reset_index()
                    counts.columns = ['Ingress Event', 'Count']
                    st.dataframe(counts.head(10), use_container_width=True)
                    fig_bi = px.bar(counts.head(10), x='Ingress Event', y='Count', color_discrete_sequence=['green'])
                    st.plotly_chart(fig_bi, use_container_width=True)
                else:
                    st.info("No Buy Ingresses found.")
                    
            with c2:
                st.markdown("### 🔴 Top SELL Ingresses")
                sell_ing = df_ing[df_ing['Type'] == 'High']
                if not sell_ing.empty:
                    counts = sell_ing['Event'].value_counts().reset_index()
                    counts.columns = ['Ingress Event', 'Count']
                    st.dataframe(counts.head(10), use_container_width=True)
                    fig_si = px.bar(counts.head(10), x='Ingress Event', y='Count', color_discrete_sequence=['red'])
                    st.plotly_chart(fig_si, use_container_width=True)
                else:
                    st.info("No Sell Ingresses found.")
        else:
            st.info("No major Ingress events occurred exactly on the pivot dates.")

        st.markdown("---")

        # --- 2. ASPECT ANALYSIS (UPDATED to Buy/Sell) ---
        st.subheader("Top New Aspect Triggers")
        if stats_data['Aspects']:
            df_asp = pd.DataFrame(stats_data['Aspects'])
            c3, c4 = st.columns(2)
            
            with c3:
                st.markdown("### 🟢 Top BUY Aspects")
                buy_asp = df_asp[df_asp['Type'] == 'Low']
                if not buy_asp.empty:
                    counts = buy_asp['Event'].value_counts().reset_index()
                    counts.columns = ['Aspect', 'Count']
                    st.dataframe(counts.head(10), use_container_width=True)
                    # Optional Bar Chart
                    fig_ba = px.bar(counts.head(10), x='Aspect', y='Count', color_discrete_sequence=['green'])
                    st.plotly_chart(fig_ba, use_container_width=True)
                else:
                    st.info("No Buy Aspects found.")
            
            with c4:
                st.markdown("### 🔴 Top SELL Aspects")
                sell_asp = df_asp[df_asp['Type'] == 'High']
                if not sell_asp.empty:
                    counts = sell_asp['Event'].value_counts().reset_index()
                    counts.columns = ['Aspect', 'Count']
                    st.dataframe(counts.head(10), use_container_width=True)
                    # Optional Bar Chart
                    fig_sa = px.bar(counts.head(10), x='Aspect', y='Count', color_discrete_sequence=['red'])
                    st.plotly_chart(fig_sa, use_container_width=True)
                else:
                    st.info("No Sell Aspects found.")
        else:
            st.info("No New Aspects started on these pivot dates.")