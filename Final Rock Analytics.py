import streamlit as st
import pandas as pd
import sqlite3
import os
import pydeck as pdk
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

# --- 1. PAGE CONFIG & STYLING ---
st.set_page_config(page_title="BOKA AIS Analytics", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: white !important; color: #1E1E1E !important; }
    h1 { font-size: 3rem !important; font-weight: 800 !important; color: #0072BC !important; }
    [data-testid="stMetricValue"] { font-size: 2.5rem !important; font-weight: 800 !important; color: #0072BC !important; }
    .stTable { width: 100% !important; font-size: 1.0rem !important; }
    </style>
    """, unsafe_allow_html=True)

DB_PATH = "active_vessel_data.db"

# --- 2. HELPER FUNCTIONS ---
def haversine_nm(lat1, lon1, lat2, lon2):
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return 0.0
    R = 3440.065 
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

# --- 3. ADVANCED LOGIC ENGINE ---
def process_ais_logic(df):
    df['RECEIVEDUTC'] = pd.to_datetime(df['RECEIVEDUTC'], errors='coerce')
    df = df.sort_values(['VESSEL_NAME', 'RECEIVEDUTC'])
    
    # --- ENHANCED SPEED DETECTION (Handles speedOverGround) ---
    # We look for any column that contains SOG or SPEED regardless of case
    speed_match = [c for c in df.columns if any(x in c.upper() for x in ['SOG', 'SPEED', 'VELOCITY'])]
    
    if speed_match:
        # If speedOverGround exists, it matches 'SPEED' and gets used here
        df['speed_val'] = pd.to_numeric(df[speed_match[0]], errors='coerce').fillna(0)
    else:
        df['speed_val'] = 0.0

    def get_color(s):
        if s < 3: return [255, 0, 0, 140]    # Red (Slow/Working)
        elif s < 7: return [255, 165, 0, 140] # Orange (Transition)
        else: return [0, 114, 188, 140]      # Blue (Transit)
    df['color'] = df['speed_val'].apply(get_color)

    # Port & Zone Detection
    df['LOCATION_NAME'] = df['NAME'].fillna('Unknown') if 'NAME' in df.columns else 'Unknown'
    df['is_loading'] = df['LOCATION_NAME'].astype(str).str.contains('Stevin Rock', case=False)
    df['is_offloading'] = df['LOCATION_NAME'].astype(str).str.contains('Offload|Project', case=False)
    
    # Gap analysis for stay durations
    df['ping_gap'] = df.groupby('VESSEL_NAME')['RECEIVEDUTC'].diff().dt.total_seconds() / 3600
    df['loading_duration'] = np.where(df['is_loading'], df['ping_gap'], 0)
    df['offloading_duration'] = np.where(df['is_offloading'], df['ping_gap'], 0)

    # Trip Logic
    df['departure_time'] = np.where(df['is_loading'], df['RECEIVEDUTC'], pd.NaT)
    df['load_lat'] = np.where(df['is_loading'], df['LATITUDE'], np.nan)
    df['load_lon'] = np.where(df['is_loading'], df['LONGITUDE'], np.nan)
    
    group = df.groupby('VESSEL_NAME', group_keys=False)
    df['departure_time'] = group['departure_time'].ffill()
    df['load_lat'] = group['load_lat'].ffill()
    df['load_lon'] = group['load_lon'].ffill()
    
    df['last_known_port'] = np.where(df['is_loading'], 'Loading', np.where(df['is_offloading'], 'Offloading', None))
    df['last_known_port'] = group['last_known_port'].ffill()
    df['Is_Trip_Start'] = ((df['is_offloading']) & (df['last_known_port'].shift(1) == 'Loading')).astype(int)

    # Trip Stats
    df['Cycle_Time_Hrs'] = 0.0
    df['Trip_Distance_NM'] = 0.0
    mask = (df['Is_Trip_Start'] == 1) & (df['departure_time'].notnull())
    if mask.any():
        t_now = pd.to_datetime(df.loc[mask, 'RECEIVEDUTC'])
        t_start = pd.to_datetime(df.loc[mask, 'departure_time'])
        df.loc[mask, 'Cycle_Time_Hrs'] = (t_now - t_start).dt.total_seconds() / 3600
        df.loc[mask, 'Trip_Distance_NM'] = df[mask].apply(
            lambda x: haversine_nm(x['load_lat'], x['load_lon'], x['LATITUDE'], x['LONGITUDE']), axis=1
        )
    
    df['Avg_Speed_NM_Hr'] = np.where(df['Cycle_Time_Hrs'] > 0, df['Trip_Distance_NM'] / df['Cycle_Time_Hrs'], 0)
    
    # Capacity Mapping
    cap_match = [c for c in df.columns if any(x in c.upper() for x in ['CAPACITY', 'TONS', 'VOL'])]
    df['Trip_Volume'] = df['Is_Trip_Start'] * pd.to_numeric(df[cap_match[0]], errors='coerce').fillna(0) if cap_match else 0
        
    df['Month'] = df['RECEIVEDUTC'].dt.strftime('%Y-%m')
    return df

# --- 4. DATA LOADING ---
if 'db_ready' not in st.session_state:
    st.session_state.db_ready = os.path.exists(DB_PATH)

if not st.session_state.db_ready:
    st.title("🚢 BOKA AIS Analytics Setup")
    uploaded = st.file_uploader("Upload boka_data.db", type=['db'])
    if uploaded:
        with open(DB_PATH, "wb") as f: f.write(uploaded.getbuffer())
        st.session_state.db_ready = True
        st.rerun()
    st.stop()

# --- 5. MAIN DASHBOARD ---
st.title("🚢 BOKA Rock AIS Analytics")

with st.sidebar:
    conn = sqlite3.connect(DB_PATH)
    # Check for Vessel Name Column
    cols = pd.read_sql("SELECT * FROM vessel_data LIMIT 0", conn).columns
    v_col = next((c for c in cols if 'VESSEL' in c.upper() or 'NAME' in c.upper()), cols[0])
    
    all_vessels = pd.read_sql(f"SELECT DISTINCT {v_col} FROM vessel_data", conn).iloc[:,0].dropna().tolist()
    conn.close()
    
    selected = st.multiselect("Select Vessels", sorted(all_vessels), default=all_vessels)
    if st.button("🗑️ Reset Database"):
        if os.path.exists(DB_PATH): os.remove(DB_PATH)
        st.session_state.db_ready = False
        st.rerun()

if selected:
    conn = sqlite3.connect(DB_PATH)
    placeholders = ', '.join(['?'] * len(selected))
    raw_df = pd.read_sql(f"SELECT * FROM vessel_data WHERE {v_col} IN ({placeholders})", conn, params=selected)
    conn.close()

    # Convert columns to uppercase for internal logic consistency
    raw_df.columns = [c.upper() for c in raw_df.columns]
    f_df = process_ais_logic(raw_df)

    # Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Trips", f"{int(f_df['Is_Trip_Start'].sum())}")
    c2.metric("Total Tons", f"{int(f_df['Trip_Volume'].sum()):,}")
    c3.metric("Avg Speed", f"{f_df[f_df['Avg_Speed_NM_Hr']>0]['Avg_Speed_NM_Hr'].mean():.1f} NM/h")
    c4.metric("Avg Loading", f"{f_df[f_df['loading_duration']>0]['loading_duration'].mean():.1f}h")
    c5.metric("Avg Offload", f"{f_df[f_df['offloading_duration']>0]['offloading_duration'].mean():.1f}h")

    # Monthly Trends
    st.subheader("📊 Performance Trends")
    m_stats = f_df.groupby('Month').agg({'Is_Trip_Start': 'sum', 'Trip_Volume': 'sum'}).reset_index().sort_values('Month')
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=m_stats['Month'], y=m_stats['Is_Trip_Start'], name="Trips", marker_color='#0072BC'), secondary_y=False)
    fig.add_trace(go.Scatter(x=m_stats['Month'], y=m_stats['Trip_Volume'], name="Tons", line=dict(color='#FFB300', width=4)), secondary_y=True)
    fig.update_layout(template="simple_white", height=300, showlegend=True, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Speed Map
    st.subheader("📍 Movement & Speed Heatmap")
    st.markdown("🔴 **< 3 kts** | 🟡 **3-7 kts** | 🔵 **> 7 kts**")
    view = pdk.ViewState(latitude=f_df['LATITUDE'].mean(), longitude=f_df['LONGITUDE'].mean(), zoom=8, pitch=30)
    layer = pdk.Layer("ScatterplotLayer", f_df.iloc[::20], get_position=['LONGITUDE', 'LATITUDE'], 
                      get_color='color', get_radius=250, radius_min_pixels=3)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, map_style="light"))

    # Leaderboard
    st.subheader("🏆 Vessel Ranking")
    leaderboard = f_df.groupby('VESSEL_NAME').agg({
        'Is_Trip_Start': 'sum',
        'Trip_Volume': 'sum',
        'Avg_Speed_NM_Hr': lambda x: x[x>0].mean(),
        'loading_duration': lambda x: x[x>0].sum() / max(1, f_df[f_df['VESSEL_NAME']==x.name]['Is_Trip_Start'].sum()),
        'offloading_duration': lambda x: x[x>0].sum() / max(1, f_df[f_df['VESSEL_NAME']==x.name]['Is_Trip_Start'].sum())
    }).rename(columns={'Is_Trip_Start': 'Trips', 'Trip_Volume': 'Tons', 'Avg_Speed_NM_Hr': 'Speed', 'loading_duration': 'Load (h)', 'offloading_duration': 'Offload (h)'})
    st.table(leaderboard.style.format({'Tons': '{:,.0f}', 'Speed': '{:.1f}', 'Load (h)': '{:.1f}', 'Offload (h)': '{:.1f}'}))
    
    st.download_button("📥 Download Report", leaderboard.to_csv().encode('utf-8'), "BOKA_Fleet_Summary.csv", "text/csv")
else:
    st.info("Select vessels from the sidebar to begin.")