"""
PRF Live Rainfall Tracker
Streamlit in Snowflake Application
Texas Farm Credit

v3.0 — Client Policies + Indemnity Estimates + Grid-County Separation
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from snowflake.snowpark.context import get_active_session

session = get_active_session()

# ─── Branding ───
FC_GREEN = "#5E9732"
FC_SLATE = "#5B707F"
FC_RUST  = "#9D5F58"
FC_CREAM = "#F5F1E8"
FC_AMBER = "#C4952B"

# ─── Interval Configuration ───
INTERVAL_CONFIG = {
    "625 — Jan-Feb": {
        "code": "625", "name": "Jan-Feb",
        "start": "01-01", "end": "02-28",
        "total_days": 59, "label": "January 1 – February 28"
    },
    "626 — Feb-Mar": {
        "code": "626", "name": "Feb-Mar",
        "start": "02-01", "end": "03-31",
        "total_days": 59, "label": "February 1 – March 31"
    },
}

# ─── Interval Name ↔ Code Mapping (all 11 PRF intervals) ───
INTERVAL_NAME_TO_CODE = {
    "Jan-Feb": "625", "Feb-Mar": "626", "Mar-Apr": "627",
    "Apr-May": "628", "May-Jun": "629", "Jun-Jul": "630",
    "Jul-Aug": "631", "Aug-Sep": "632", "Sep-Oct": "633",
    "Oct-Nov": "634", "Nov-Dec": "635",
}
INTERVAL_CODE_TO_NAME = {v: k for k, v in INTERVAL_NAME_TO_CODE.items()}


# ═══════════════════════════════════════════
# STYLES
# ═══════════════════════════════════════════

st.markdown(f"""
<style>
    .main .block-container,
    .appview-container .main .block-container,
    section[data-testid="stMainBlockContainer"] {{
        max-width: 100% !important;
        width: 100% !important;
        padding: 1rem 2rem !important;
    }}
    div[data-testid="stMetric"] {{
        background: {FC_CREAM};
        border: 2px solid {FC_GREEN};
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }}
    div[data-testid="stMetric"] label {{ 
        color: {FC_SLATE} !important; 
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{ 
        color: #2d3a2e !important; 
        font-size: 2rem !important;
        font-weight: 800 !important;
    }}
    .signal-indemnity {{ 
        background: {FC_GREEN}; color: white; padding: 8px 20px; 
        border-radius: 8px; font-weight: 700; font-size: 1rem;
        text-align: center; margin-top: 4px;
    }}
    .signal-ok {{ 
        background: {FC_SLATE}; color: white; padding: 8px 20px; 
        border-radius: 8px; font-weight: 700; font-size: 1rem;
        text-align: center; margin-top: 4px;
    }}
    .indemnity-card {{
        border-radius: 10px; padding: 20px; text-align: center;
        font-weight: 700; color: white;
    }}
    section[data-testid="stSidebar"] {{ min-width: 320px; max-width: 380px; }}
    div[data-testid="stDataFrame"],
    div[data-testid="stDataFrame"] > div {{
        width: 100% !important;
    }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# CALCULATION FUNCTIONS (Decimal arithmetic — matches CvC app)
# ═══════════════════════════════════════════

def round_half_up(value, decimals=2):
    """Round using 'round half up' to match PRF official tool."""
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return 0.0
    d = Decimal(str(value))
    if decimals == 0:
        quantize_to = Decimal('1')
    else:
        quantize_to = Decimal('0.' + '0' * (decimals - 1) + '1')
    return float(d.quantize(quantize_to, rounding=ROUND_HALF_UP))


def calculate_shortfall_decimal(trigger_level, index_value):
    """Calculate shortfall using Decimal to avoid IEEE 754 errors."""
    trigger_dec = Decimal(str(trigger_level))
    index_dec = Decimal(str(index_value))
    if index_dec >= trigger_dec:
        return Decimal('0')
    return (trigger_dec - index_dec) / trigger_dec


def calculate_protection_decimal(county_base_value, coverage_level, productivity_factor, insurable_interest=1.0):
    """Calculate dollar protection per acre with Decimal precision."""
    cbv  = Decimal(str(county_base_value))
    cov  = Decimal(str(coverage_level))
    prod = Decimal(str(productivity_factor))
    ins  = Decimal(str(insurable_interest))
    result = cbv * cov * prod * ins
    return float(result.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


def calculate_position_indemnity(final_index, coverage_level, county_base_value,
                                  productivity_factor, insurable_interest,
                                  acres, allocation_pct):
    """
    Calculate indemnity for one position for one interval.
    Matches CvC backtest Decimal arithmetic exactly.
    
    Args:
        final_index: Projected rainfall index value (0-150+)
        coverage_level: As decimal (e.g. 0.75)
        county_base_value: Dollar CBV
        productivity_factor: e.g. 1.35
        insurable_interest: e.g. 1.0
        acres: Total insured acres for this position
        allocation_pct: Percent allocated to this interval (e.g. 28 for 28%)
    
    Returns:
        Integer indemnity payment in dollars
    """
    trigger = coverage_level * 100  # 0.75 → 75.0

    if final_index >= trigger:
        return 0

    # Protection per acre (CBV × coverage × productivity × insurable interest)
    protection_per_acre = calculate_protection_decimal(
        county_base_value, coverage_level, productivity_factor, insurable_interest
    )
    total_protection = protection_per_acre * acres

    # Interval protection = Round(total_protection × allocation%)
    alloc_dec = Decimal(str(allocation_pct)) / Decimal('100')
    total_prot_dec = Decimal(str(total_protection))
    interval_protection_dec = (total_prot_dec * alloc_dec).quantize(
        Decimal('1'), rounding=ROUND_HALF_UP
    )
    interval_protection = int(interval_protection_dec)

    # Shortfall and indemnity
    shortfall_dec = calculate_shortfall_decimal(trigger, final_index)
    raw_indemnity = shortfall_dec * Decimal(str(interval_protection))

    if raw_indemnity >= Decimal('0.01'):
        return int(raw_indemnity.quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    return 0


# ═══════════════════════════════════════════
# CACHED DATA LOADERS
# ═══════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_texas_grids(interval_code):
    return session.sql(f"""
        WITH grid_counties AS (
            SELECT TRY_TO_NUMBER(m.SUB_COUNTY_CODE) AS GRID_ID,
                   LISTAGG(DISTINCT c.COUNTY_NAME, ' / ') WITHIN GROUP (ORDER BY c.COUNTY_NAME) AS COUNTY_NAME
            FROM MAP_YTD m
            LEFT JOIN COUNTY_YTD c 
                ON c.STATE_CODE = '48' AND c.COUNTY_CODE = m.COUNTY_CODE
                AND c.REINSURANCE_YEAR = 2025 AND c.DELETED_DATE IS NULL
            WHERE m.INSURANCE_PLAN_CODE = '13' AND m.STATE_CODE = '48' AND m.DELETED_DATE IS NULL
            GROUP BY 1
        )
        SELECT n.GRID_ID, n.NORMAL_IN, n.CV_PCT, n.CONFIDENCE_TIER,
               gc.COUNTY_NAME, g.CENTER_LAT, g.CENTER_LON
        FROM PRF_GRID_NORMALS n
        JOIN grid_counties gc ON n.GRID_ID = gc.GRID_ID
        LEFT JOIN PRF_GRID_REFERENCE g ON g.GRIDCODE = n.GRID_ID
        WHERE n.INTERVAL_CODE = '{interval_code}'
        ORDER BY n.GRID_ID
    """).to_pandas()


@st.cache_data(ttl=600)
def load_rainfall(date_start, date_end):
    """Uses PRF_GRID_CPC_MAP for correct coordinate snapping."""
    return session.sql(f"""
        SELECT 
            m.GRID_ID,
            ROUND(SUM(r.PRECIP_IN), 4) AS RAIN_SO_FAR,
            COUNT(DISTINCT r.OBSERVATION_DATE) AS DAYS_COLLECTED,
            MAX(r.OBSERVATION_DATE) AS LAST_DAY,
            MIN(r.FILE_TYPE) AS FILE_TYPE
        FROM PRF_RAINFALL_REALTIME r
        JOIN PRF_GRID_CPC_MAP m
            ON ROUND(r.LATITUDE, 3) = m.CPC_LAT
            AND ROUND(r.LONGITUDE, 3) = m.CPC_LON
        WHERE r.OBSERVATION_DATE BETWEEN '{date_start}' AND '{date_end}'
        GROUP BY 1
    """).to_pandas()


@st.cache_data(ttl=3600)
def load_client_list():
    """Load available client policies."""
    try:
        df = session.sql("""
            SELECT CLIENT_NAME, CROP_YEAR, POLICY_VERSION,
                   ARRAY_SIZE(POLICY_JSON:positions) AS POSITION_COUNT,
                   POLICY_JSON:defaults:coverage_level::FLOAT AS COVERAGE_LEVEL
            FROM PRF_CLIENT_POLICIES
            ORDER BY CLIENT_NAME, CROP_YEAR DESC
        """).to_pandas()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_client_policy(client_name, crop_year):
    """Load and flatten a client's policy JSON into a positions DataFrame."""
    df = session.sql(f"""
        SELECT
            p.value:grid_id::INT              AS GRID_ID,
            p.value:county::VARCHAR           AS COUNTY,
            p.value:state::VARCHAR            AS STATE,
            p.value:acres::INT                AS ACRES,
            p.value:county_base_value::FLOAT  AS COUNTY_BASE_VALUE,
            p.value:allocation                AS ALLOCATION_JSON,
            pol.POLICY_JSON:defaults:coverage_level::FLOAT       AS COVERAGE_LEVEL,
            pol.POLICY_JSON:defaults:productivity_factor::FLOAT  AS PRODUCTIVITY_FACTOR,
            pol.POLICY_JSON:defaults:insurable_interest::FLOAT   AS INSURABLE_INTEREST
        FROM PRF_CLIENT_POLICIES pol,
             LATERAL FLATTEN(input => pol.POLICY_JSON:positions) p
        WHERE pol.CLIENT_NAME = '{client_name}'
          AND pol.CROP_YEAR = {crop_year}
    """).to_pandas()

    # Parse allocation JSON string into Python dicts
    import json
    def parse_alloc(val):
        if isinstance(val, str):
            return json.loads(val)
        elif isinstance(val, dict):
            return val
        return {}

    df['ALLOCATION'] = df['ALLOCATION_JSON'].apply(parse_alloc)
    return df


# ═══════════════════════════════════════════
# TRACKER & GAUGE FUNCTIONS
# ═══════════════════════════════════════════

def build_tracker(grids_df, rain_df, coverage_level, total_days):
    merged = grids_df.merge(rain_df, on="GRID_ID", how="inner")
    merged["PARTIAL_INDEX"] = (merged["RAIN_SO_FAR"] / merged["NORMAL_IN"] * 100).round(1)
    merged["DAILY_RATE"] = merged["RAIN_SO_FAR"] / merged["DAYS_COLLECTED"]
    merged["PROJECTED_RAIN"] = (merged["DAILY_RATE"] * total_days).round(4)
    merged["PROJECTED_INDEX"] = (merged["PROJECTED_RAIN"] / merged["NORMAL_IN"] * 100).round(1)
    trigger = coverage_level
    merged["SIGNAL"] = merged["PROJECTED_INDEX"].apply(
        lambda idx: "LIKELY INDEMNITY" if idx < trigger else "OK"
    )
    return merged


def create_gauge(grid_id, projected_index, partial_index, signal,
                 rain_so_far, normal_in, days, total_days,
                 coverage_level, county_name=None):

    bar_color = FC_GREEN if signal == "LIKELY INDEMNITY" else FC_SLATE
    trigger = coverage_level
    max_range = max(150, projected_index + 20)
    pct_through = round(days / total_days * 100)

    county_str = ""
    if county_name and pd.notna(county_name):
        county_str = f"  ·  {county_name}"

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=projected_index,
        number={
            "valueformat": ".1f",
            "font": {"size": 64, "color": "#2d3a2e", "family": "Arial Black"},
        },
        title={
            "text": (
                f"<b style='font-size:20px'>Grid {grid_id}</b>"
                f"<span style='font-size:13px;color:{FC_SLATE}'>{county_str}</span>"
                f"<br>"
                f"<span style='font-size:14px;color:{FC_SLATE}'>"
                f"Rain: <b>{rain_so_far:.2f}\"</b> of {normal_in:.1f}\" normal"
                f"  ·  {days}/{total_days} days ({pct_through}%)"
                f"  ·  Coverage: {coverage_level}%"
                f"</span>"
            ),
            "font": {"size": 14, "color": "#2d3a2e"},
        },
        gauge={
            "axis": {
                "range": [0, max_range], "tickwidth": 2,
                "tickcolor": FC_SLATE, "tickfont": {"color": FC_SLATE, "size": 14},
                "dtick": 25,
            },
            "bar": {"color": bar_color, "thickness": 0.75},
            "bgcolor": "#e8e4dd", "borderwidth": 0,
            "steps": [
                {"range": [0, trigger], "color": "rgba(94, 151, 50, 0.28)"},
                {"range": [trigger, max_range], "color": "rgba(91, 112, 127, 0.08)"},
            ],
            "threshold": {
                "line": {"color": "#2d3a2e", "width": 5},
                "thickness": 0.9, "value": partial_index,
            },
        },
        domain={"x": [0.15, 0.85], "y": [0, 0.85]},
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=100, b=10),
        paper_bgcolor=FC_CREAM, plot_bgcolor=FC_CREAM,
        font={"color": "#2d3a2e"},
    )
    return fig


# ═══════════════════════════════════════════
# INDEMNITY SCENARIO ENGINE
# ═══════════════════════════════════════════

def compute_indemnity_scenarios(positions_df, rain_df, normals_df, interval_name, total_days):
    """
    For each client position with allocation for the given interval,
    compute three indemnity scenarios: pessimistic, current trend, optimistic.
    
    Returns a DataFrame with one row per position.
    """
    rows = []

    for _, pos in positions_df.iterrows():
        alloc = pos['ALLOCATION']
        alloc_pct = alloc.get(interval_name, 0)
        if alloc_pct == 0:
            continue  # Position has no allocation for this interval

        grid_id = pos['GRID_ID']

        # Look up rainfall for this grid
        rain_row = rain_df[rain_df['GRID_ID'] == grid_id]
        if rain_row.empty:
            rain_so_far = 0.0
            days_collected = 0
        else:
            rain_so_far = float(rain_row.iloc[0]['RAIN_SO_FAR'])
            days_collected = int(rain_row.iloc[0]['DAYS_COLLECTED'])

        # Look up normal for this grid
        normal_row = normals_df[normals_df['GRID_ID'] == grid_id]
        if normal_row.empty:
            continue  # Can't compute without normal
        normal_in = float(normal_row.iloc[0]['NORMAL_IN'])
        if normal_in <= 0:
            continue

        # Current partial index
        partial_index = round((rain_so_far / normal_in) * 100, 1)

        # ─── Three Scenarios ───
        # From PRF policyholder perspective: more payout = better
        # Optimistic = highest payout (zero rain remaining)
        # Pessimistic = lowest payout (normal pace remaining)
        remaining_days = max(0, total_days - days_collected)

        # Optimistic (for PRF): zero rain remaining → max shortfall → max payout
        optimistic_rain = rain_so_far
        optimistic_index = round((optimistic_rain / normal_in) * 100, 1)

        # Current trend: linear extrapolation
        if days_collected > 0:
            daily_rate = rain_so_far / days_collected
            trend_rain = round(daily_rate * total_days, 4)
        else:
            trend_rain = 0.0
        trend_index = round((trend_rain / normal_in) * 100, 1)

        # Pessimistic (for PRF): remaining days at normal rate → min shortfall → min payout
        normal_daily_rate = normal_in / total_days
        pessimistic_rain = rain_so_far + (normal_daily_rate * remaining_days)
        pessimistic_index = round((pessimistic_rain / normal_in) * 100, 1)

        # ─── Dollar Indemnities ───
        cov   = pos['COVERAGE_LEVEL']
        cbv   = pos['COUNTY_BASE_VALUE']
        pf    = pos['PRODUCTIVITY_FACTOR']
        ii    = pos['INSURABLE_INTEREST']
        acres = pos['ACRES']

        protection_per_acre = calculate_protection_decimal(cbv, cov, pf, ii)
        total_protection = protection_per_acre * acres

        indem_pessimistic = calculate_position_indemnity(
            pessimistic_index, cov, cbv, pf, ii, acres, alloc_pct)
        indem_trend = calculate_position_indemnity(
            trend_index, cov, cbv, pf, ii, acres, alloc_pct)
        indem_optimistic = calculate_position_indemnity(
            optimistic_index, cov, cbv, pf, ii, acres, alloc_pct)

        rows.append({
            'GRID_ID': grid_id,
            'COUNTY': pos['COUNTY'],
            'ACRES': acres,
            'ALLOC_PCT': alloc_pct,
            'CBV': cbv,
            'NORMAL_IN': normal_in,
            'RAIN_SO_FAR': rain_so_far,
            'DAYS': days_collected,
            'PARTIAL_INDEX': partial_index,
            'PESSIMISTIC_IDX': pessimistic_index,
            'TREND_IDX': trend_index,
            'OPTIMISTIC_IDX': optimistic_index,
            'PROTECTION': round(total_protection * alloc_pct / 100, 0),
            'INDEM_PESSIMISTIC': indem_pessimistic,
            'INDEM_TREND': indem_trend,
            'INDEM_OPTIMISTIC': indem_optimistic,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ═══════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════

with st.sidebar:
    st.markdown(f"### ⚙️ Controls")

    # ── INTERVAL SELECTOR ──
    interval_selection = st.selectbox(
        "📅 Interval",
        options=list(INTERVAL_CONFIG.keys()),
        index=0,
        help="Select the coverage interval to track"
    )
    interval = INTERVAL_CONFIG[interval_selection]

    # ── CLIENT POLICY SELECTOR ──
    st.divider()
    st.markdown("**📋 Client Policy**")

    client_list_df = load_client_list()
    client_mode = False
    client_positions_df = None
    policy_coverage = None

    if not client_list_df.empty:
        client_options = ["— Manual —"] + [
            f"{r['CLIENT_NAME']} ({int(r['CROP_YEAR'])})"
            for _, r in client_list_df.iterrows()
        ]
        client_selection = st.selectbox("Load Client", client_options, index=0)

        if client_selection != "— Manual —":
            client_mode = True
            # Parse selection back to name + year
            sel_row = client_list_df.iloc[client_options.index(client_selection) - 1]
            client_name = sel_row['CLIENT_NAME']
            crop_year = int(sel_row['CROP_YEAR'])
            policy_coverage = float(sel_row['COVERAGE_LEVEL'])

            client_positions_df = load_client_policy(client_name, crop_year)

            # Filter to positions with allocation for selected interval
            interval_name = interval['name']  # e.g. "Jan-Feb"
            mask = client_positions_df['ALLOCATION'].apply(
                lambda a: a.get(interval_name, 0) > 0
            )
            client_positions_df = client_positions_df[mask].copy()

            pos_ct = len(client_positions_df)
            total_ac = client_positions_df['ACRES'].sum() if pos_ct > 0 else 0
            st.caption(
                f"✅ **{client_name}** — {pos_ct} positions for {interval_name} "
                f"· {total_ac:,} acres"
            )
    else:
        st.caption("No client policies found. Using manual mode.")

    st.divider()

    # ── COVERAGE LEVEL ──
    cov_options = [90, 85, 80, 75, 70]
    if client_mode and policy_coverage:
        # Map policy decimal to display integer (0.75 → 75)
        policy_cov_int = int(policy_coverage * 100)
        default_idx = cov_options.index(policy_cov_int) if policy_cov_int in cov_options else 0
    else:
        default_idx = 0

    coverage_level = st.selectbox(
        "Coverage Level",
        options=cov_options,
        index=default_idx,
        help="Indemnity triggers below this index level"
    )

    st.divider()

    # ── GRID SELECTION (manual mode only) ──
    if not client_mode:
        grids_df = load_texas_grids(interval["code"])

        grids_df["LABEL"] = grids_df.apply(
            lambda r: f"{r['GRID_ID']} — {r['COUNTY_NAME']}"
            if pd.notna(r.get("COUNTY_NAME")) else str(r["GRID_ID"]), axis=1
        )
        label_to_id = dict(zip(grids_df["LABEL"], grids_df["GRID_ID"]))

        all_counties = set()
        for names in grids_df["COUNTY_NAME"].dropna():
            for c in names.split(" / "):
                all_counties.add(c.strip())

        selected_counties = st.multiselect("Filter by County", sorted(all_counties), default=[])

        if selected_counties:
            mask = grids_df["COUNTY_NAME"].apply(
                lambda x: any(c in str(x) for c in selected_counties) if pd.notna(x) else False
            )
            filtered_labels = grids_df[mask]["LABEL"].tolist()
        else:
            filtered_labels = grids_df["LABEL"].tolist()

        st.markdown("**Select Grids**")
        grid_entry = st.text_input(
            "Enter Grid IDs (comma separated)",
            placeholder="7929, 8230, 8231",
            help="Type grid IDs directly"
        )
        selected_labels = st.multiselect("Or pick from list", filtered_labels, default=[])

        st.divider()
        st.markdown("**Quick Select**")
        col1, col2 = st.columns(2)
        with col1:
            top_n = st.selectbox("Driest N", [10, 25, 50, "All"], index=0)
        with col2:
            show_all_likely = st.checkbox("Likely only", value=False)
    else:
        # In client mode, load grids for gauge display
        grids_df = load_texas_grids(interval["code"])
        grid_entry = None
        selected_labels = []
        top_n = 10
        show_all_likely = False

    st.divider()
    generate = st.button("🚀 Generate", type="primary", use_container_width=True)

    st.divider()
    st.markdown(f"""
    <div style='font-size:11px;color:{FC_SLATE}'>
        <b>Sources:</b> CPC Gauge (RT) · 3yr normals · Linear projection<br>
        <b>Interval:</b> {interval['code']} ({interval['label']})
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════

st.markdown("# 🌾 PRF Live Rainfall Tracker")
st.markdown(
    f"**{interval['name']} 2026 · Interval {interval['code']}** "
    f"· Real-time CPC rainfall vs 3-year implied normals"
)
st.divider()


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

if generate:

    date_start = f"2026-{interval['start']}"
    date_end   = f"2026-{interval['end']}"
    total_days = interval["total_days"]
    interval_name = interval["name"]

    rain_df = load_rainfall(date_start, date_end)
    tracker = build_tracker(grids_df, rain_df, coverage_level, total_days)

    # ─────────────────────────────────────
    # CLIENT MODE
    # ─────────────────────────────────────
    if client_mode and client_positions_df is not None and not client_positions_df.empty:

        # Get the unique grid IDs from client positions
        client_grid_ids = client_positions_df['GRID_ID'].unique().tolist()
        display_df = tracker[tracker["GRID_ID"].isin(client_grid_ids)].copy()

        if display_df.empty:
            st.warning("No rainfall data found for client grids. Check pipeline status.")
            st.stop()

        display_df = display_df.sort_values("PROJECTED_INDEX", ascending=True)
        days_in = int(display_df["DAYS_COLLECTED"].iloc[0]) if len(display_df) > 0 else 0
        likely_ct = len(display_df[display_df["SIGNAL"] == "LIKELY INDEMNITY"])

        # ── Metrics Row ──
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Positions", len(client_positions_df))
        c2.metric("Days", f"{days_in} / {total_days}")
        c3.metric("Coverage", f"{coverage_level}%")
        c4.metric("Likely Indemnity", likely_ct)

        # ──────────────────────────────────
        # INDEMNITY ESTIMATES
        # ──────────────────────────────────
        st.divider()
        st.markdown("### 💰 Portfolio Indemnity Estimates")

        indem_df = compute_indemnity_scenarios(
            client_positions_df, rain_df, grids_df, interval_name, total_days
        )

        if not indem_df.empty:
            total_pessimistic = int(indem_df['INDEM_PESSIMISTIC'].sum())
            total_trend       = int(indem_df['INDEM_TREND'].sum())
            total_optimistic  = int(indem_df['INDEM_OPTIMISTIC'].sum())

            # ── Scenario Cards ──
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.markdown(f"""
                <div class="indemnity-card" style="background:{FC_GREEN};">
                    <div style="font-size:13px; opacity:0.9;">OPTIMISTIC</div>
                    <div style="font-size:11px; opacity:0.7;">Zero rain remaining</div>
                    <div style="font-size:32px; margin-top:8px;">${total_optimistic:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            with sc2:
                st.markdown(f"""
                <div class="indemnity-card" style="background:{FC_AMBER};">
                    <div style="font-size:13px; opacity:0.9;">CURRENT TREND</div>
                    <div style="font-size:11px; opacity:0.7;">Linear extrapolation</div>
                    <div style="font-size:32px; margin-top:8px;">${total_trend:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            with sc3:
                st.markdown(f"""
                <div class="indemnity-card" style="background:{FC_RUST};">
                    <div style="font-size:13px; opacity:0.9;">PESSIMISTIC</div>
                    <div style="font-size:11px; opacity:0.7;">Normal pace remaining</div>
                    <div style="font-size:32px; margin-top:8px;">${total_pessimistic:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")

            # ── Position Breakdown Table ──
            st.markdown("#### Position Breakdown")

            # Build pre-formatted string table to prevent Streamlit arrow indicators
            tbl = pd.DataFrame()
            tbl['Grid']       = indem_df['GRID_ID'].astype(str)
            tbl['County']     = indem_df['COUNTY']
            tbl['Acres']      = indem_df['ACRES'].apply(lambda x: f"{x:,}")
            tbl['Alloc %']    = indem_df['ALLOC_PCT'].apply(lambda x: f"{x:.0f}%")
            tbl['CBV']        = indem_df['CBV'].apply(lambda x: f"${x:.2f}")
            tbl['Normal']     = indem_df['NORMAL_IN'].apply(lambda x: f"{x:.2f}\"")
            tbl['Rain']       = indem_df['RAIN_SO_FAR'].apply(lambda x: f"{x:.2f}\"")
            tbl['Days']       = indem_df['DAYS'].astype(str)
            tbl['Cur Idx']    = indem_df['PARTIAL_INDEX'].apply(lambda x: f"{x:.1f}")
            tbl['Opt Idx']    = indem_df['OPTIMISTIC_IDX'].apply(lambda x: f"{x:.1f}")
            tbl['Trend Idx']  = indem_df['TREND_IDX'].apply(lambda x: f"{x:.1f}")
            tbl['Pess Idx']   = indem_df['PESSIMISTIC_IDX'].apply(lambda x: f"{x:.1f}")
            tbl['Protection'] = indem_df['PROTECTION'].apply(lambda x: f"${x:,.0f}")
            tbl['$ Optimistic']  = indem_df['INDEM_OPTIMISTIC'].apply(lambda x: f"${x:,.0f}")
            tbl['$ Trend']       = indem_df['INDEM_TREND'].apply(lambda x: f"${x:,.0f}")
            tbl['$ Pessimistic'] = indem_df['INDEM_PESSIMISTIC'].apply(lambda x: f"${x:,.0f}")

            # Totals row
            totals = pd.DataFrame([{
                'Grid': 'TOTAL',
                'County': '',
                'Acres': f"{int(indem_df['ACRES'].sum()):,}",
                'Alloc %': '',
                'CBV': '',
                'Normal': '',
                'Rain': '',
                'Days': '',
                'Cur Idx': '',
                'Opt Idx': '',
                'Trend Idx': '',
                'Pess Idx': '',
                'Protection': f"${int(indem_df['PROTECTION'].sum()):,}",
                '$ Optimistic': f"${int(indem_df['INDEM_OPTIMISTIC'].sum()):,}",
                '$ Trend': f"${int(indem_df['INDEM_TREND'].sum()):,}",
                '$ Pessimistic': f"${int(indem_df['INDEM_PESSIMISTIC'].sum()):,}",
            }])
            tbl = pd.concat([tbl, totals], ignore_index=True)

            st.dataframe(
                tbl, use_container_width=True, hide_index=True,
                height=min(600, 50 + len(tbl) * 40),
            )
        else:
            st.info(f"No client positions have allocation for {interval_name}.")

        # ── Gauges ──
        st.divider()
        st.markdown("### 📊 Projected Final Index")

        st.markdown(f"""
        <div style="
            display: flex; align-items: center; gap: 32px; 
            padding: 12px 20px; background: {FC_CREAM}; 
            border: 1px solid #d5d0c6; border-radius: 8px; margin-bottom: 16px;
            flex-wrap: wrap;
        ">
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="width:40px; height:14px; background:{FC_GREEN}; border-radius:3px;"></div>
                <span style="font-size:13px; color:#2d3a2e;"><b>Bar</b> — Projected Final Index</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="width:4px; height:22px; background:#2d3a2e; border-radius:1px;"></div>
                <span style="font-size:13px; color:#2d3a2e;"><b>Line</b> — Current Estimated Index Value</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="width:40px; height:14px; background:rgba(94,151,50,0.28); border:1px solid #ccc; border-radius:3px;"></div>
                <span style="font-size:13px; color:#2d3a2e;">Indemnity zone (below {coverage_level})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        for _, row in display_df.iterrows():
            fig = create_gauge(
                grid_id=row["GRID_ID"],
                projected_index=row["PROJECTED_INDEX"],
                partial_index=row["PARTIAL_INDEX"],
                signal=row["SIGNAL"],
                rain_so_far=row["RAIN_SO_FAR"],
                normal_in=row["NORMAL_IN"],
                days=row["DAYS_COLLECTED"],
                total_days=total_days,
                coverage_level=coverage_level,
                county_name=row.get("COUNTY_NAME"),
            )
            st.plotly_chart(fig, use_container_width=True)

            sig = row["SIGNAL"]
            proj = row["PROJECTED_INDEX"]
            part = row["PARTIAL_INDEX"]
            if sig == "LIKELY INDEMNITY":
                st.markdown(
                    f'<div class="signal-indemnity">'
                    f'✅ LIKELY INDEMNITY — Current: {part:.1f}  ·  Projected: {proj:.1f}  ·  Trigger: {coverage_level}'
                    f'</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="signal-ok">'
                    f'OK — Current: {part:.1f}  ·  Projected: {proj:.1f}'
                    f'</div>', unsafe_allow_html=True)

            st.markdown("")

    # ─────────────────────────────────────
    # MANUAL MODE (existing behavior)
    # ─────────────────────────────────────
    else:
        display_df = None
        if grid_entry and grid_entry.strip():
            try:
                typed_ids = [int(x.strip()) for x in grid_entry.split(",") if x.strip()]
                display_df = tracker[tracker["GRID_ID"].isin(typed_ids)].copy()
            except ValueError:
                st.error("Invalid grid IDs. Use comma-separated numbers like: 7929, 8230")
                st.stop()
        elif selected_labels:
            selected_ids = [label_to_id[lbl] for lbl in selected_labels]
            display_df = tracker[tracker["GRID_ID"].isin(selected_ids)].copy()
        elif show_all_likely:
            display_df = tracker[tracker["SIGNAL"] == "LIKELY INDEMNITY"].copy()
        else:
            n = len(tracker) if top_n == "All" else int(top_n)
            display_df = tracker.nsmallest(n, "PROJECTED_INDEX").copy()

        if display_df is None or display_df.empty:
            st.warning("No grids found.")
            st.stop()

        display_df = display_df.sort_values("PROJECTED_INDEX", ascending=True)

        days_in = int(display_df["DAYS_COLLECTED"].iloc[0])
        likely_ct = len(display_df[display_df["SIGNAL"] == "LIKELY INDEMNITY"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Grids", len(display_df))
        c2.metric("Days", f"{days_in} / {total_days}")
        c3.metric("Coverage", f"{coverage_level}%")
        c4.metric("Likely Indemnity", likely_ct)

        st.divider()
        st.markdown("### 📊 Projected Final Index")

        # ── Legend ──
        st.markdown(f"""
        <div style="
            display: flex; align-items: center; gap: 32px; 
            padding: 12px 20px; background: {FC_CREAM}; 
            border: 1px solid #d5d0c6; border-radius: 8px; margin-bottom: 16px;
            flex-wrap: wrap;
        ">
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="width:40px; height:14px; background:{FC_GREEN}; border-radius:3px;"></div>
                <span style="font-size:13px; color:#2d3a2e;"><b>Bar</b> — Projected Final Index</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="width:4px; height:22px; background:#2d3a2e; border-radius:1px;"></div>
                <span style="font-size:13px; color:#2d3a2e;"><b>Line</b> — Current Estimated Index Value</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="width:40px; height:14px; background:rgba(94,151,50,0.28); border:1px solid #ccc; border-radius:3px;"></div>
                <span style="font-size:13px; color:#2d3a2e;">Indemnity zone (below {coverage_level})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        for _, row in display_df.iterrows():
            fig = create_gauge(
                grid_id=row["GRID_ID"],
                projected_index=row["PROJECTED_INDEX"],
                partial_index=row["PARTIAL_INDEX"],
                signal=row["SIGNAL"],
                rain_so_far=row["RAIN_SO_FAR"],
                normal_in=row["NORMAL_IN"],
                days=row["DAYS_COLLECTED"],
                total_days=total_days,
                coverage_level=coverage_level,
                county_name=row.get("COUNTY_NAME"),
            )
            st.plotly_chart(fig, use_container_width=True)

            sig = row["SIGNAL"]
            proj = row["PROJECTED_INDEX"]
            part = row["PARTIAL_INDEX"]
            if sig == "LIKELY INDEMNITY":
                st.markdown(
                    f'<div class="signal-indemnity">'
                    f'✅ LIKELY INDEMNITY — Current: {part:.1f}  ·  Projected: {proj:.1f}  ·  Trigger: {coverage_level}'
                    f'</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="signal-ok">'
                    f'OK — Current: {part:.1f}  ·  Projected: {proj:.1f}'
                    f'</div>', unsafe_allow_html=True)

            st.markdown("")

        st.divider()
        st.markdown("### 📋 Detail")

        table_df = display_df[[
            "GRID_ID", "COUNTY_NAME", "NORMAL_IN", "DAYS_COLLECTED",
            "RAIN_SO_FAR", "PARTIAL_INDEX", "PROJECTED_RAIN",
            "PROJECTED_INDEX", "SIGNAL", "CV_PCT"
        ]].copy()
        table_df.columns = [
            "Grid", "Counties", "Normal (in)", "Days",
            "Rain (in)", "Current Idx", "Proj Rain (in)",
            "Proj Index", "Signal", "CV%"
        ]

        st.dataframe(
            table_df, use_container_width=True, hide_index=True,
            height=min(600, 50 + len(table_df) * 40),
            column_config={
                "Grid": st.column_config.NumberColumn(format="%d"),
                "Normal (in)": st.column_config.NumberColumn(format="%.1f"),
                "Rain (in)": st.column_config.NumberColumn(format="%.1f"),
                "Proj Rain (in)": st.column_config.NumberColumn(format="%.1f"),
                "Current Idx": st.column_config.NumberColumn(format="%.1f"),
                "Proj Index": st.column_config.NumberColumn(format="%.1f"),
                "CV%": st.column_config.NumberColumn(format="%.1f"),
            }
        )

else:
    st.markdown(f"""
    <div style='text-align:center; padding:80px 20px; color:{FC_SLATE};'>
        <h2 style='color:{FC_GREEN};'>Select grids and click Generate</h2>
        <p style='font-size:16px;'>
            Choose your interval and coverage level.<br>
            Load a client policy or select grids manually.<br>
            Then click Generate.
        </p>
        <p style='font-size:14px; margin-top:20px;'>
            🌾 Insured grids · 3-year implied normals · 0.4% avg CV · RT data through today
        </p>
    </div>
    """, unsafe_allow_html=True)
