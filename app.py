import streamlit as st
import pandas as pd
import numpy as np
import traceback
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

import pickle
import os

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def _save_cache(dfs, kpis_results):
    path = os.path.join(CACHE_DIR, "session_cache.pkl")
    with open(path, "wb") as f:
        pickle.dump({"dataframes": dfs, **kpis_results}, f)

def _load_cache():
    path = os.path.join(CACHE_DIR, "session_cache.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

from pipeline.data_loader import load_all
from pipeline.analysis import (
    exec_kpis, exec_insights,
    workforce_kpis, workforce_insights,
    customer_kpis, customer_insights,
    financial_kpis, financial_insights,
    compute_correlations,
)
from pipeline.ml_models import train_models
from pipeline.preprocessing import clean_for_ml
from utils.session import init as init_session

st.set_page_config(page_title="Service Analytics", page_icon="📊", layout="wide")
init_session()

# ──────────────────────────────────────
#  ADVANCED STYLING
# ──────────────────────────────────────

def _theme_css():
    dark = st.session_state.theme == "dark"
    if dark:
        bg = "linear-gradient(135deg, #0f1729 0%, #1a1f35 50%, #121828 100%)"
        card_bg = "linear-gradient(145deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%)"
        card_border = "rgba(255,255,255,0.06)"
        text_primary = "#f0f0f5"
        text_secondary = "rgba(255,255,255,0.45)"
        text_muted = "rgba(255,255,255,0.25)"
        glass_bg = "rgba(255,255,255,0.03)"
        glass_border = "rgba(255,255,255,0.06)"
        section_bg = "rgba(46,204,113,0.04)"
        section_border = "rgba(46,204,113,0.2)"
        sidebar_bg = "linear-gradient(180deg, rgba(15,23,41,0.98) 0%, rgba(20,28,52,0.98) 100%)"
        sidebar_border = "rgba(46,204,113,0.08)"
        upload_bg = "linear-gradient(135deg, rgba(46,204,113,0.06) 0%, rgba(52,152,219,0.04) 100%)"
        upload_border = "rgba(255,255,255,0.06)"
    else:
        bg = "linear-gradient(135deg, #f8f9fa 0%, #e9ecef 50%, #f1f3f5 100%)"
        card_bg = "linear-gradient(145deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.85) 100%)"
        card_border = "rgba(0,0,0,0.08)"
        text_primary = "#1a1a2e"
        text_secondary = "rgba(0,0,0,0.5)"
        text_muted = "rgba(0,0,0,0.3)"
        glass_bg = "rgba(255,255,255,0.7)"
        glass_border = "rgba(0,0,0,0.06)"
        section_bg = "rgba(46,204,113,0.06)"
        section_border = "rgba(46,204,113,0.3)"
        sidebar_bg = "linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(245,247,250,0.98) 100%)"
        sidebar_border = "rgba(0,0,0,0.06)"
        upload_bg = "linear-gradient(135deg, rgba(46,204,113,0.08) 0%, rgba(52,152,219,0.06) 100%)"
        upload_border = "rgba(0,0,0,0.08)"

    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}
    .main > div {{ padding: 0 1.5rem; }}
    .stApp {{ background: {bg}; }}
    h1, h2, h3 {{ color: {text_primary} !important; font-weight: 600 !important; letter-spacing: -0.03em !important; }}
    h1 {{ font-size: 2rem !important; }}
    h2 {{ font-size: 1.3rem !important; }}
    h3 {{ font-size: 1.05rem !important; }}

    .glass {{
        background: {glass_bg};
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid {glass_border};
        border-radius: 16px;
        padding: 24px;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    .glass:hover {{
        border-color: rgba(46,204,113,0.15);
        box-shadow: 0 8px 40px rgba(11,65,153,0.08);
    }}

    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 12px;
        margin: 8px 0 24px 0;
    }}
    .kpi-card {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-radius: 14px;
        padding: 18px 14px;
        text-align: center;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }}
    .kpi-card::before {{
        content: '';
        position: absolute; top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #2ecc71, transparent);
        opacity: 0;
        transition: opacity 0.35s;
    }}
    .kpi-card:hover::before {{ opacity: 1; }}
    .kpi-card:hover {{
        transform: translateY(-3px);
        border-color: rgba(46,204,113,0.2);
        box-shadow: 0 12px 40px rgba(52,152,219,0.12);
    }}
    .kpi-icon {{ font-size: 1.3rem; margin-bottom: 4px; }}
    .kpi-val {{
        font-size: 1.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2ecc71, #3498db);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.03em;
        line-height: 1.2;
    }}
    .kpi-lbl {{
        font-size: 0.68rem;
        color: {text_secondary};
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 500;
        margin-top: 6px;
    }}
    .kpi-sub {{
        font-size: 0.6rem;
        color: {text_muted};
        margin-top: 2px;
    }}

    .page-title {{
        display: flex; flex-direction: column; align-items: center; gap: 2px;
        margin: 0 0 6px 0;
        padding: 20px 0 14px 0;
        border-bottom: 1px solid {card_border};
        text-align: center;
    }}
    .page-title h1 {{ margin: 0; font-size: 1.6rem; }}
    .page-title .sub {{
        color: {text_secondary};
        font-size: 0.85rem;
        letter-spacing: 0.3px;
        margin-top: 2px;
    }}

    .section-title {{
        color: {text_primary};
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.3px;
        margin: 24px 0 12px 0;
        padding: 10px 16px;
        border-bottom: 2px solid {section_border};
        text-align: center;
        background: {section_bg};
        border-radius: 8px 8px 0 0;
    }}

    .stButton > button {{
        background: linear-gradient(135deg, #2ecc71, #2980b9);
        border: none;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 12px 32px;
        border-radius: 10px;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        color: white;
        letter-spacing: 0.3px;
        position: relative;
        overflow: hidden;
    }}
    .stButton > button::after {{
        content: '';
        position: absolute; inset: 0;
        background: linear-gradient(135deg, transparent 40%, rgba(255,255,255,0.08) 100%);
        pointer-events: none;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(41,128,185,0.3);
    }}

    section[data-testid="stSidebar"] {{
        background: {sidebar_bg} !important;
        border-right: 1px solid {sidebar_border};
    }}
    section[data-testid="stSidebar"] .stButton > button {{
        width: 250px;
        height: 50px;
        padding: 12px 20px;
        border-radius: 10px;
        font-size: 0.85rem;
        text-align: center;
        justify-content: center;
        display: flex;
        align-items: center;
        gap: 8px;
        letter-spacing: 0.2px;
        margin: 0 auto;
    }}

    .upload-section {{
        background: {upload_bg};
        border-radius: 24px;
        padding: 40px 32px 28px;
        border: 1.5px solid {upload_border};
        margin: 20px 0;
        transition: all 0.3s;
        backdrop-filter: blur(8px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.06);
    }}
    .upload-section:hover {{
        border-color: rgba(46,204,113,0.2);
        box-shadow: 0 8px 40px rgba(46,204,113,0.06);
    }}
    .upload-section div[data-testid="stFileUploader"] {{
        text-align: center;
    }}
    .upload-section div[data-testid="stFileUploader"] label {{
        color: {text_primary} !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        display: block !important;
        text-align: center !important;
        padding: 4px 0;
    }}
    .upload-section div[data-testid="stFileUploader"] small {{
        color: {text_secondary} !important;
        font-size: 0.75rem;
    }}
    .upload-section div[data-testid="stFileUploader"] button {{
        margin: 0 auto;
    }}
    .upload-title {{
        font-size: 1.3rem; font-weight: 700;
        background: linear-gradient(135deg, #2ecc71, #3498db);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        margin: 8px 0 2px 0;
    }}
    .upload-sub {{
        color: {text_secondary};
        font-size: 0.8rem;
    }}

    .stDataFrame {{
        border: 1px solid {card_border};
        border-radius: 12px;
        overflow: hidden;
    }}
    div[data-testid="stMetricValue"] {{
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #2ecc71, #3498db);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    div[data-testid="stMetricLabel"] {{
        font-size: 0.72rem !important;
        color: {text_secondary} !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    .chart-wrap {{
        background: {glass_bg};
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid {glass_border};
        border-radius: 14px;
        padding: 12px;
        margin: 8px 0;
        transition: all 0.3s;
    }}
    .chart-wrap:hover {{ border-color: rgba(46,204,113,0.12); }}

    hr {{
        border-color: {card_border} !important;
        margin: 20px 0 !important;
    }}

    .stSpinner > div > div {{
        border-color: #2ecc71 transparent transparent transparent !important;
    }}
    div[data-testid="stNotification"] {{
        background: {card_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 12px !important;
    }}
    .main {{
        animation: fadeIn 0.5s ease-out;
    }}

    @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(12px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    @media (max-width: 768px) {{
        .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
        .main > div {{ padding: 0 0.75rem; }}
    }}
</style>
"""

st.markdown(_theme_css(), unsafe_allow_html=True)

# ──────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────

LABELS = {
    "orders":"Orders","users":"Users","workers":"Workers",
    "ratings":"Ratings","categories":"Categories",
}
REQUIRED = list(LABELS.keys())
ICONS = {
    "orders":"📋","users":"👥","workers":"👷",
    "ratings":"⭐","categories":"🏷️",
}

PAGES = [
    ("📈  Executive Dashboard", "exec"),
    ("👷  Workforce & Services", "workforce"),
    ("👥  Customer & Order Funnel", "customer"),
    ("💰  Financial & Payment", "financial"),
    ("🤖  ML Models", "ml"),
    ("🔗  Correlations", "corr"),
]

# ──────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────

def _theme_colors():
    dark = st.session_state.theme == "dark"
    return {
        "font": "rgba(255,255,255,0.7)" if dark else "rgba(0,0,0,0.65)",
        "grid": "rgba(255,255,255,0.04)" if dark else "rgba(0,0,0,0.06)",
        "line": "rgba(255,255,255,0.3)" if dark else "rgba(0,0,0,0.15)",
        "title": "rgba(255,255,255,0.9)" if dark else "rgba(0,0,0,0.8)",
    }

def wrap_chart(fig, title=None):
    dark = st.session_state.theme == "dark"
    font_color = "rgba(255,255,255,0.7)" if dark else "rgba(0,0,0,0.65)"
    title_color = "rgba(255,255,255,0.85)" if dark else "rgba(0,0,0,0.8)"
    grid_color = "rgba(255,255,255,0.04)" if dark else "rgba(0,0,0,0.06)"
    upd = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=font_color, size=11),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        legend=dict(font=dict(size=10), orientation="h", y=1.1),
    )
    if title:
        upd["title"] = dict(text=title, font=dict(size=14, color=title_color))
    fig.update_layout(**upd)
    fig.update_xaxes(gridcolor=grid_color, zeroline=False)
    fig.update_yaxes(gridcolor=grid_color, zeroline=False)
    st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_kpis(items):
    html = '<div class="kpi-grid">'
    for icon, val, lbl, sub in items:
        html += f"""
        <div class='kpi-card'>
            <div class='kpi-icon'>{icon}</div>
            <div class='kpi-val'>{val}</div>
            <div class='kpi-lbl'>{lbl}</div>
            {f'<div class="kpi-sub">{sub}</div>' if sub else ''}
        </div>"""
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def page_header(title, subtitle):
    st.markdown(f"""
    <div class='page-title'>
        <h1>{title}</h1>
        <span class='sub'>{subtitle}</span>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────
#  UPLOAD PAGE
# ──────────────────────────────────────

def upload_page():
    st.markdown("""
    <div style="text-align:center; margin: 16px auto 10px; max-width:520px;">
        <div style="font-size:2.8rem; margin-bottom:4px;">📂</div>
        <div class="upload-title">Upload Your Data — رفع البيانات</div>
        <div class="upload-sub">Upload all <strong>5 CSV files</strong> below to start the analysis</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    cols = st.columns(3)
    files = {}
    for i, name in enumerate(REQUIRED):
        with cols[i % 3]:
            u = st.file_uploader(f"{ICONS.get(name,'')} **{LABELS[name]}**", type="csv", key=f"f_{name}")
            if u:
                files[name] = u.getvalue()
    st.markdown("</div>", unsafe_allow_html=True)
    if len(files) == 5:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if st.button("🚀 Analyze & Launch Dashboard — تحليل وبدء", use_container_width=True):
                with st.spinner("Loading, analyzing, and training ..."):
                    try:
                        dfs = load_all(files)
                        st.session_state.dataframes = dfs
                        st.session_state.exec_kpis = exec_kpis(dfs)
                        st.session_state.exec_insights = exec_insights(dfs)
                        st.session_state.workforce_kpis = workforce_kpis(dfs)
                        st.session_state.workforce_insights = workforce_insights(dfs)
                        st.session_state.customer_kpis = customer_kpis(dfs)
                        st.session_state.customer_insights = customer_insights(dfs)
                        st.session_state.financial_kpis = financial_kpis(dfs)
                        st.session_state.financial_insights = financial_insights(dfs)
                        st.session_state.correlations = compute_correlations(dfs)
                        st.session_state.ml_results = train_models(dfs)
                        st.session_state.loaded = True
                        st.session_state._user_chose_upload = False
                        _save_cache(dfs, {
                            "exec_kpis": st.session_state.exec_kpis,
                            "exec_insights": st.session_state.exec_insights,
                            "workforce_kpis": st.session_state.workforce_kpis,
                            "workforce_insights": st.session_state.workforce_insights,
                            "customer_kpis": st.session_state.customer_kpis,
                            "customer_insights": st.session_state.customer_insights,
                            "financial_kpis": st.session_state.financial_kpis,
                            "financial_insights": st.session_state.financial_insights,
                            "correlations": st.session_state.correlations,
                            "ml_results": st.session_state.ml_results,
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.code(traceback.format_exc())


# ──────────────────────────────────────
#  FILTER HELPERS
# ──────────────────────────────────────

def _date_filter_key(page):
    return f"f_{page}_date"

def _filter_orders(dfs, page):
    """Return a copy of dfs with orders filtered by date range for a page."""
    st.session_state.setdefault("filters", {})
    orders = dfs.get("orders")
    if orders is None:
        return dict(dfs)

    fkey = _date_filter_key(page)
    dr = st.session_state.filters.get(fkey, (None, None))

    out = dict(dfs)
    filtered = orders.copy()
    if dr[0]:
        filtered = filtered[filtered["created_at"] >= pd.Timestamp(dr[0])]
    if dr[1]:
        filtered = filtered[filtered["created_at"] <= pd.Timestamp(dr[1])]
    out["orders"] = filtered
    return out


def _filter_bar(page):
    """Render filter bar for a page. Returns True if filters are active."""
    st.session_state.setdefault("filters", {})
    orders = st.session_state.dataframes.get("orders")
    if orders is None or "created_at" not in orders.columns:
        return

    fkey = _date_filter_key(page)
    current = st.session_state.filters.get(fkey, (None, None))
    min_d = orders["created_at"].min()
    max_d = orders["created_at"].max()
    if pd.isna(min_d) or pd.isna(max_d):
        return

    cols = st.columns([3, 1])
    with cols[0]:
        dr = st.date_input(
            "📅 Date Range — الفترة الزمنية",
            value=(current[0] or min_d.date(), current[1] or max_d.date()),
            min_value=min_d.date(), max_value=max_d.date(),
            key=f"date_{page}", label_visibility="collapsed",
        )
    with cols[1]:
        if st.button("✕ Clear", key=f"clear_{page}"):
            st.session_state.filters[fkey] = (None, None)
            st.rerun()

    if isinstance(dr, tuple) and len(dr) == 2:
        st.session_state.filters[fkey] = dr
    else:
        st.session_state.filters[fkey] = (None, None)


# ──────────────────────────────────────
#  PAGE 1 — EXECUTIVE DASHBOARD
# ──────────────────────────────────────

def page_exec():
    dfs = _filter_orders(st.session_state.dataframes, "exec")
    k = exec_kpis(dfs)
    page_header("📈 Executive Dashboard", "نظرة عامة على أداء المنصة")
    _filter_bar("exec")
    render_kpis([
        ("📋", k.get("total_orders",0), "Total Orders", "إجمالي الطلبات"),
        ("👷", k.get("active_workers",0), "Active Workers", f"عاملين نشطين · {k.get('active_workers_rate',0)}%"),
        ("⭐", f"{k.get('avg_worker_rating',0)}/5", "Avg Worker Rating", "تقييم العاملين"),
        ("✅", f"{k.get('completion_rate',0)}%", "Completion Rate", "معدل الإنجاز"),
        ("❌", f"{k.get('cancellation_rate',0)}%", "Cancellation Rate", "معدل الإلغاء"),
        ("📈", f"{k.get('customer_growth_rate',0)}%", "Customer Growth", "نمو العملاء"),
    ])
    st.divider()
    ins = exec_insights(dfs)

    # ── 1 & 2. Best Services + Province Distribution ──
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='section-title'>🏆 Best Services — الخدمات الأعلى طلباً</div>", unsafe_allow_html=True)
        bs = ins.get("best_services")
        if bs:
            df = pd.DataFrame(bs)
            fig = px.bar(df, x="orders", y="service", orientation="h",
                         title="",
                         color="orders", color_continuous_scale="greens",
                         text="orders")
            fig.update_traces(textposition="outside")
            fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
            wrap_chart(fig)
    with col_b:
        st.markdown("<div class='section-title'>📍 Province Distribution — توزيع العملاء حسب المحافظة</div>", unsafe_allow_html=True)
        ps = ins.get("province_stats")
        if ps:
            df = pd.DataFrame(ps)
            cols = ins.get("province_cols", df.columns.tolist())
            st.dataframe(df, hide_index=True, use_container_width=True, height=500)

    # ── 3. Process Pie ──
    st.markdown("<div class='section-title'>🔄 Order Status Flow — توزيع حالات الطلبات</div>", unsafe_allow_html=True)
    pp = ins.get("process_pie")
    if pp:
        fig = px.pie(values=list(pp.values()), names=list(pp.keys()),
                     hole=0.45, title="",
                     color_discrete_sequence=["#2ecc71", "#e74c3c", "#f39c12", "#3498db"])
        wrap_chart(fig)

    # ── 4. Satisfaction Trend ──
    st.markdown("<div class='section-title'>📈 Satisfaction Trend — اتجاه رضا العملاء</div>", unsafe_allow_html=True)
    sts = ins.get("satisfaction_trend")
    if sts:
        df = pd.DataFrame(sts)
        fig = px.line(df, x="month", y="avg_rating", markers=True,
                      title="",
                      range_y=[0, 5])
        fig.add_hline(y=4.0, line_dash="dash", line_color="rgba(46,204,113,0.4)",
                      annotation_text="Target 4.0", annotation_position="bottom right")
        fig.update_traces(line=dict(width=2.5, color="#2ecc71"),
                          fill="tozeroy", fillcolor="rgba(46,204,113,0.08)")
        wrap_chart(fig)


# ──────────────────────────────────────
#  PAGE 2 — WORKFORCE & SERVICES
# ──────────────────────────────────────

def page_workforce():
    page_header("👷 Workforce & Services Performance", "أداء العاملين وفئات الخدمة")
    _filter_bar("workforce")

    dfs = _filter_orders(st.session_state.dataframes, "workforce")
    orders = dfs.get("orders")

    # Extra filters
    cat_options = ["All"]
    if orders is not None:
        cat_col = next((c for c in orders.columns if c.lower() == "category"), None)
        if cat_col:
            cat_options += sorted(orders[cat_col].dropna().unique())
    svc_sel = st.session_state.filters.setdefault("f_workforce_service", "All")
    rat_sel = st.session_state.filters.setdefault("f_workforce_rating", 0.0)

    cols = st.columns([2, 2])
    with cols[0]:
        svc_sel = st.selectbox("🏷️ Service — الخدمة", cat_options,
                               index=cat_options.index(svc_sel) if svc_sel in cat_options else 0,
                               key="wf_svc")
    with cols[1]:
        rat_sel = st.slider("⭐ Min Rating — أقل تقييم", 0.0, 5.0, rat_sel, 0.5, key="wf_rat")

    st.session_state.filters["f_workforce_service"] = svc_sel
    st.session_state.filters["f_workforce_rating"] = rat_sel

    if svc_sel != "All" and orders is not None:
        cat_col = next((c for c in orders.columns if c.lower() == "category"), None)
        if cat_col:
            orders = orders[orders[cat_col] == svc_sel]
    if rat_sel > 0 and orders is not None:
        rat = dfs.get("ratings")
        if rat is not None:
            rwu = next((c for c in rat.columns if c.lower() == "worker_username"), None)
            rstar = next((c for c in rat.columns if c.lower() == "stars"), None)
            if rwu and rstar:
                good = rat[rat[rstar] >= rat_sel][rwu].unique()
                owu = next((c for c in orders.columns if c.lower() == "worker_username"), None)
                if owu:
                    orders = orders[orders[owu].isin(good)]
    dfs["orders"] = orders
    # Re-derive related tables from filtered orders
    if orders is not None:
        owu = next((c for c in orders.columns if c.lower() == "worker_username"), None)
        if owu:
            wus = set(orders[owu].dropna().unique())
            for tbl, col in [("ratings","worker_username"),("workers","username")]:
                src = st.session_state.dataframes.get(tbl)
                if src is not None and col in src.columns:
                    dfs[tbl] = src[src[col].isin(wus)]

    k = workforce_kpis(dfs)
    render_kpis([
        ("👷", k.get("total_workers",0), "Total Workers", "إجمالي العاملين"),
        ("🔥", f"{k.get('active_workers_rate',0)}%", "Active Workers Rate", "نسبة النشاط"),
        ("📦", f"{k.get('avg_orders_per_worker',0)}", "Avg Orders/Worker", "متوسط الطلبات لكل عامل"),
        ("⭐", f"{k.get('avg_worker_rating',0)}/5", "Avg Worker Rating", "متوسط التقييم"),
        ("🏷️", k.get("top_category","N/A"), "Top Service", "الخدمة الأعلى طلباً"),
        ("💰", f"${k.get('revenue_per_worker',0):,.0f}", "Revenue/Worker", "العائد لكل عامل"),
    ])
    st.divider()
    ins = workforce_insights(dfs)

    # ── Profession Overview (dual Y) ──
    st.markdown("<div class='section-title'>📊 Profession Overview — نظرة شاملة على المهن</div>", unsafe_allow_html=True)
    po = ins.get("profession_overview")
    if po:
        df = pd.DataFrame(po)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["profession"], y=df["worker_count"],
                             name="👷 Workers", marker_color="rgba(46,204,113,0.7)",
                             yaxis="y"))
        fig.add_trace(go.Scatter(x=df["profession"], y=df["avg_experience"],
                                 name="🎓 Experience (yrs)", mode="lines+markers",
                                 marker=dict(color="#3498db", size=8),
                                 line=dict(color="#3498db", width=2.5),
                                 yaxis="y2"))
        fig.add_trace(go.Scatter(x=df["profession"], y=df["avg_rating"],
                                 name="⭐ Rating (1-5)", mode="lines+markers",
                                 marker=dict(color="#f39c12", size=8),
                                 line=dict(color="#f39c12", width=2.5),
                                 yaxis="y3"))
        fig.update_layout(
            yaxis=dict(title="👷 Workers", side="left", gridcolor=_theme_colors()["grid"]),
            yaxis2=dict(title="🎓 Experience (yrs)", overlaying="y", side="right",
                        gridcolor=_theme_colors()["grid"]),
            yaxis3=dict(title="⭐ Rating (1-5)", overlaying="y", position=0.95, side="right",
                        gridcolor=_theme_colors()["grid"]),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=_theme_colors()["font"], size=11),
            legend=dict(orientation="h", y=1.15),
        )
        wrap_chart(fig)

    # ── Worker Availability + Acceptance Rate (2 cols) ──
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.markdown("<div class='section-title'>✅ Worker Availability — توفر العاملين</div>", unsafe_allow_html=True)
        wa = ins.get("worker_availability")
        if wa:
            df = pd.DataFrame(wa)
            df["status"] = df["is_available"].astype(str).map({"True": "Available", "False": "Unavailable"})
            fig = px.bar(df, x="profession", y="count", color="status",
                         barmode="group", title="",
                         color_discrete_map={"Available": "#2ecc71", "Unavailable": "#e74c3c"})
            wrap_chart(fig)
    with col_q2:
        st.markdown("<div class='section-title'>📈 Acceptance Rate — نسبة القبول</div>", unsafe_allow_html=True)
        ar = ins.get("acceptance_rate")
        if ar:
            df = pd.DataFrame(ar)
            fig = px.bar(df, x="profession", y="avg_accept_rate",
                         color="avg_accept_rate", color_continuous_scale="greens",
                         text="avg_accept_rate", title="")
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(yaxis=dict(range=[0, 100]))
            wrap_chart(fig)

    # ── Experience vs Rating ──
    st.markdown("<div class='section-title'>🎓 Experience vs Rating — الخبرة والتقييم لكل مهنة</div>", unsafe_allow_html=True)
    evr = ins.get("exp_vs_rating")
    if evr:
        df = pd.DataFrame(evr)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["profession"], y=df["avg_experience"],
                             name="🎓 Experience (yrs)",
                             marker_color="rgba(52,152,219,0.8)", yaxis="y"))
        fig.add_trace(go.Bar(x=df["profession"], y=df["avg_rating"],
                             name="⭐ Rating (1-5)",
                             marker_color="rgba(46,204,113,0.8)", yaxis="y2"))
        fig.update_layout(
            yaxis=dict(title="🎓 Years", side="left", gridcolor=_theme_colors()["grid"]),
            yaxis2=dict(title="⭐ Rating", overlaying="y", side="right",
                        gridcolor=_theme_colors()["grid"]),
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=_theme_colors()["font"], size=11),
            legend=dict(orientation="h", y=1.15),
        )
        wrap_chart(fig)

    # ── 3. Worker Utilization Gap ──
    st.markdown("<div class='section-title'>⚡ Utilization Gap — فجوة استغلال العمال</div>", unsafe_allow_html=True)
    cvd = ins.get("capacity_vs_demand")
    if cvd:
        df = pd.DataFrame(cvd)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["service"], y=df["workers"],
                             name="👷 Workers Available",
                             marker_color="rgba(46,204,113,0.7)",
                             yaxis="y"))
        fig.add_trace(go.Bar(x=df["service"], y=df["orders"],
                             name="📦 Orders Demand",
                             marker_color="rgba(52,152,219,0.7)",
                             yaxis="y2"))
        fig.add_trace(go.Scatter(x=df["service"], y=df["ratio"],
                                 name="📊 Orders/Worker Ratio",
                                 mode="lines+markers",
                                 marker=dict(color="#f39c12", size=8),
                                 line=dict(color="#f39c12", width=2.5),
                                 yaxis="y3"))
        fig.update_layout(
            title="Worker Capacity vs Orders Demand",
            yaxis=dict(title="👷 Workers", side="left",
                       gridcolor=_theme_colors()["grid"]),
            yaxis2=dict(title="📦 Orders", side="right",
                        overlaying="y", gridcolor=_theme_colors()["grid"]),
            yaxis3=dict(title="📊 Ratio", overlaying="y", position=0.95,
                        side="right", gridcolor=_theme_colors()["grid"]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=_theme_colors()["font"], size=11),
            legend=dict(orientation="h", y=1.15),
        )
        wrap_chart(fig)

    # ── 4. Service Revenue & Cancellation ──
    st.markdown("<div class='section-title'>💰 Revenue & Cancellation — الإيرادات والإلغاء حسب الخدمة</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    src = ins.get("service_revenue_cancel")
    if src:
        df = pd.DataFrame(src)
        cat_col = next((c for c in df.columns if c.lower() == "category"), "service")
        fig = px.bar(df, x=cat_col, y=["total_orders", "cancelled"],
                     title="📦 Orders & Cancellations by Service",
                     barmode="group",
                     color_discrete_map={"total_orders": "#2ecc71", "cancelled": "#ff6b6b"})
        with c1:
            wrap_chart(fig)
        if "revenue" in df.columns:
            fig2 = px.bar(df, x=cat_col, y="revenue",
                          color="cancel_rate",
                          title="💰 Revenue & ❌ Cancellation Rate",
                          color_continuous_scale="RdYlGn_r",
                          text=df["cancel_rate"].apply(lambda x: f"{x}%"))
            fig2.update_traces(textposition="outside")
            with c2:
                wrap_chart(fig2)

    # ── 5. Service Demand (sorted bar) ──
    st.markdown("<div class='section-title'>📈 Service Demand — الطلب على الخدمات</div>", unsafe_allow_html=True)
    sd = ins.get("most_requested")
    if sd:
        df = pd.DataFrame(sd).sort_values("orders", ascending=True)
        fig = px.bar(df, x="orders", y="service", orientation="h",
                     color="orders", color_continuous_scale="greens",
                     text="orders")
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        wrap_chart(fig)

    # ── 6. Quality Trend ──
    st.markdown("<div class='section-title'>📊 Quality Trends — اتجاهات الجودة الشهرية</div>", unsafe_allow_html=True)
    qt = ins.get("quality_trend")
    if qt:
        df = pd.DataFrame(qt)
        fig = px.line(df, x="month", y="avg_rating", markers=True,
                      title="Average Rating Over Time",
                      range_y=[0, 5],
                      color_discrete_sequence=["#9b59b6"])
        fig.update_traces(line=dict(width=2.5),
                          fill="tozeroy", fillcolor="rgba(155,89,182,0.08)")
        wrap_chart(fig)


# ──────────────────────────────────────
#  PAGE 3 — CUSTOMER & ORDER FUNNEL
# ──────────────────────────────────────

def page_customer():
    page_header("👥 Customer & Order Funnel", "تحليل سلوك العملاء والطلبات")
    _filter_bar("customer")

    dfs = _filter_orders(st.session_state.dataframes, "customer")
    orders = dfs.get("orders")

    # Extra filters
    st_options = ["All", "COMPLETED", "CANCELLED", "PENDING", "WAITING"]
    st_sel = st.session_state.filters.setdefault("f_customer_status", "All")
    gov_sel = st.session_state.filters.setdefault("f_customer_gov", "All")

    users = st.session_state.dataframes.get("users")
    gov_options = ["All"]
    if users is not None:
        gov_col = next((c for c in users.columns if c.lower() == "governorate"), None)
        if gov_col:
            gov_options += sorted(users[gov_col].dropna().unique())

    cols = st.columns([2, 2])
    with cols[0]:
        st_sel = st.selectbox("📌 Status — الحالة", st_options,
                              index=st_options.index(st_sel) if st_sel in st_options else 0,
                              key="cu_st")
    with cols[1]:
        gov_sel = st.selectbox("📍 Governorate — المحافظة", gov_options,
                               index=gov_options.index(gov_sel) if gov_sel in gov_options else 0,
                               key="cu_gov")

    st.session_state.filters["f_customer_status"] = st_sel
    st.session_state.filters["f_customer_gov"] = gov_sel

    if st_sel != "All" and orders is not None:
        sc = next((c for c in orders.columns if c.lower() == "status"), None)
        if sc:
            orders = orders[orders[sc].str.upper() == st_sel]
    if gov_sel != "All" and orders is not None and users is not None:
        cu = next((c for c in orders.columns if c.lower() == "client_username"), None)
        ugov = next((c for c in users.columns if c.lower() == "governorate"), None)
        uuname = next((c for c in users.columns if c.lower() == "username"), None)
        if cu and ugov and uuname:
            gov_clients = set(users[users[ugov] == gov_sel][uuname].unique())
            orders = orders[orders[cu].isin(gov_clients)]
    dfs["orders"] = orders
    if orders is not None:
        cu = next((c for c in orders.columns if c.lower() == "client_username"), None)
        if cu:
            cus = set(orders[cu].dropna().unique())
            src = st.session_state.dataframes.get("users")
            uname = next((c for c in src.columns if c.lower() == "username"), None) if src is not None else None
            if src is not None and uname:
                dfs["users"] = src[src[uname].isin(cus)]

    k = customer_kpis(dfs)
    render_kpis([
        ("👥", k.get("total_clients",0), "Total Clients", "إجمالي العملاء"),
        ("🔄", f"{k.get('repeat_rate',0)}%", "Repeat Rate", "نسبة العملاء المتكررين"),
        ("📦", f"{k.get('avg_orders_per_client',0)}", "Avg Orders/Client", "متوسط الطلبات لكل عميل"),
        ("⏱️", f"{k.get('avg_fulfillment_hours',0)}h", "Avg Fulfillment Time", "متوسط وقت التنفيذ"),
        ("❌", f"{k.get('cancellation_rate',0)}%", "Cancellation Rate", "معدل الإلغاء"),
    ])
    st.divider()
    ins = customer_insights(dfs)

    # ── 1. Order Funnel ──
    st.markdown("<div class='section-title'>🧩 Order Funnel — تدفق الطلبات عبر المراحل</div>", unsafe_allow_html=True)
    funnel = ins.get("order_funnel")
    if funnel:
        colors = ["#2ecc71", "#e74c3c", "#f39c12", "#3498db", "#95a5a6"][:len(funnel)]
        fig = go.Figure(go.Funnel(
            y=list(funnel.keys()),
            x=list(funnel.values()),
            textposition="inside",
            textinfo="value+percent initial",
            marker=dict(
                color=colors,
                line=dict(width=2, color="white")),
        ))
        fig.update_layout(title="Order Funnel",
                          paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color=_theme_colors()["font"], size=11))
        wrap_chart(fig)

    # ── 3. Orders by Governorate / Peak Hours Heatmap ──
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("<div class='section-title'>🗺️ Orders by Governorate — توزيع الطلبات حسب المحافظة</div>", unsafe_allow_html=True)
        geo = ins.get("geo_orders")
        if geo:
            df = pd.DataFrame(geo)
            fig = px.bar(df, x="orders", y="governorate", orientation="h",
                         color="orders", color_continuous_scale="blues")
            fig.update_layout(yaxis=dict(autorange="reversed"))
            wrap_chart(fig)
    with col_d:
        st.markdown("<div class='section-title'>⏰ Peak Hours Heatmap — أوقات الذروة (ساعة × يوم)</div>", unsafe_allow_html=True)
        heat = ins.get("peak_hours_heatmap")
        if heat:
            df = pd.DataFrame(heat)
            fig = px.density_heatmap(df, x="hour", y="dow", z="orders",
                                     color_continuous_scale="greens",
                                     range_x=[0, 23])
            fig.update_layout(xaxis=dict(tickmode="linear", dtick=2),
                              yaxis=dict(categoryorder="array",
                                         categoryarray=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]))
            wrap_chart(fig)

    # ── 3. Cancelled Orders by Service ──
    st.markdown("<div class='section-title'>📉 Cancelled Orders by Service — الطلبات الملغية حسب الخدمة</div>", unsafe_allow_html=True)
    col_g, col_h = st.columns(2)
    with col_g:
        cbc = ins.get("cancelled_by_category")
        if cbc:
            df = pd.DataFrame(cbc)
            fig = px.bar(df, x="category", y=["total", "cancelled"],
                         title="Total vs Cancelled by Service",
                         barmode="group",
                         color_discrete_map={"total": "#2ecc71", "cancelled": "#ff6b6b"})
            wrap_chart(fig)
    with col_h:
        cbc = ins.get("cancelled_by_category")
        if cbc:
            df = pd.DataFrame(cbc)
            fig2 = px.bar(df, x="category", y="cancel_rate",
                          title="Cancellation Rate % by Service",
                          color="cancel_rate", color_continuous_scale="RdYlGn_r",
                          text="cancel_rate")
            fig2.update_traces(texttemplate="%{text}%", textposition="outside")
            wrap_chart(fig2)

    # ── 4. Satisfaction by Service ──
    st.markdown("<div class='section-title'>⭐ Satisfaction by Service — تقييم العملاء لكل خدمة</div>", unsafe_allow_html=True)
    sbs = ins.get("satisfaction_by_service")
    if sbs:
        df = pd.DataFrame(sbs)
        fig = px.bar(df, x="service", y="avg_rating",
                     title="Average Customer Rating by Service",
                     color="avg_rating", color_continuous_scale="RdYlGn",
                     range_y=[0, 5], text="avg_rating")
        fig.update_traces(texttemplate="%{text}/5", textposition="outside")
        wrap_chart(fig)


# ──────────────────────────────────────
#  PAGE 4 — FINANCIAL & PAYMENT
# ──────────────────────────────────────

def page_financial():
    page_header("💰 Financial & Payment Analysis", "التحليل المالي وعمليات الدفع")
    _filter_bar("financial")

    dfs = _filter_orders(st.session_state.dataframes, "financial")
    orders = dfs.get("orders")

    # Extra filters
    cat_options = ["All"]
    if orders is not None:
        cat_col = next((c for c in orders.columns if c.lower() == "category"), None)
        if cat_col:
            cat_options += sorted(orders[cat_col].dropna().unique())
    svc_sel = st.session_state.filters.setdefault("f_financial_service", "All")

    cols = st.columns([2, 2])
    with cols[0]:
        svc_sel = st.selectbox("🏷️ Service — الخدمة", cat_options,
                               index=cat_options.index(svc_sel) if svc_sel in cat_options else 0,
                               key="fi_svc")

    st.session_state.filters["f_financial_service"] = svc_sel

    if svc_sel != "All" and orders is not None:
        cat_col = next((c for c in orders.columns if c.lower() == "category"), None)
        if cat_col:
            orders = orders[orders[cat_col] == svc_sel]
    dfs["orders"] = orders

    k = financial_kpis(dfs)
    growth = k.get("monthly_growth", 0)
    growth_str = f"+{growth}%" if growth > 0 else f"{growth}%"
    growth_icon = "📈" if growth > 0 else "📉"
    render_kpis([
        ("💰", f"${k.get('total_collected',0):,.0f}", "Total Collected", "إجمالي المحصل"),
        ("💵", f"${k.get('avg_transaction_value',0):,.2f}", "Avg Commission", "متوسط العمولة"),
        ("📊", f"{k.get('effective_rate',0)}%", "Effective Rate", "نسبة العمولة الفعلية"),
        ("📈", f"${k.get('avg_daily_commission',0):,.2f}", "Avg Daily Commission", "متوسط العمولة اليومي"),
        ("💎", f"${k.get('max_transaction',0):,.2f}", "Max Commission", "أعلى عمولة"),
        (growth_icon, growth_str, "Monthly Growth", "نمو الإيرادات الشهري"),
    ])
    st.divider()
    ins = financial_insights(dfs)

    # ── 1. Revenue & Commission Trends ──
    st.markdown("<div class='section-title'>📈 Revenue & Commission Trends — اتجاهات الإيرادات والعمولات</div>", unsafe_allow_html=True)
    rct = ins.get("revenue_commission_trends")
    if rct:
        df = pd.DataFrame(rct)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["month"], y=df["revenue"],
                                 name="💰 Commission",
                                 mode="lines+markers",
                                 marker=dict(color="#2ecc71", size=8),
                                 line=dict(color="#2ecc71", width=3),
                                 yaxis="y"))
        fig.update_layout(
            title="Monthly Commission Revenue",
            yaxis=dict(title="💰 Commission ($)", side="left",
                       gridcolor=_theme_colors()["grid"]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=_theme_colors()["font"], size=11),
            legend=dict(orientation="h", y=1.15),
        )
        wrap_chart(fig)

    # ── 3. Top 5 Earning Workers ──
    st.markdown("<div class='section-title'>👷 Top 5 Earning Workers — أعلى 5 عاملين أرباحاً</div>", unsafe_allow_html=True)
    tew = ins.get("top_earning_workers")
    if tew:
        df = pd.DataFrame(tew)
        name_col = "name" if "name" in df.columns else "username"
        fig = px.bar(df, x="total_commission", y=name_col, orientation="h",
                     title="",
                     color="total_commission", color_continuous_scale="greens",
                     text="total_commission")
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
        wrap_chart(fig)

    # ── 4 & 5. Revenue by Service + Revenue by Governorate ──
    st.markdown("<div class='section-title'>💰 Revenue by Service & Governorate — الإيرادات حسب الخدمة والمحافظة</div>", unsafe_allow_html=True)
    col_c, col_d = st.columns(2)
    with col_c:
        rbs = ins.get("revenue_by_service")
        if rbs:
            df = pd.DataFrame(rbs)
            fig = px.bar(df, x="service", y="revenue",
                         title="",
                         color="revenue", color_continuous_scale="greens",
                         text="revenue")
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            wrap_chart(fig)
    with col_d:
        rbg = ins.get("revenue_by_governorate")
        if rbg:
            df = pd.DataFrame(rbg)
            fig = px.bar(df, x="revenue", y="governorate", orientation="h",
                         title="",
                         color="revenue", color_continuous_scale="blues",
                         text="revenue")
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            fig.update_layout(yaxis=dict(autorange="reversed"))
            wrap_chart(fig)

    # ── 6. Commission by Day of Week ──
    st.markdown("<div class='section-title'>📅 Commission by Day of Week — العمولة حسب أيام الأسبوع</div>", unsafe_allow_html=True)
    cbd = ins.get("commission_by_dow")
    if cbd:
        df = pd.DataFrame(cbd)
        fig = px.bar(df, x="day", y="commission",
                     color="commission", color_continuous_scale="greens",
                     text="commission")
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_layout(xaxis=dict(categoryorder="array",
                                     categoryarray=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]))
        wrap_chart(fig)



# ──────────────────────────────────────
#  PAGE 5 — ML MODELS
# ──────────────────────────────────────

def page_ml():
    dfs = _filter_orders(st.session_state.dataframes, "ml")
    ml = train_models(dfs)
    page_header("🤖 Machine Learning Models", "النماذج التنبؤية والتحليلات الذكية")
    _filter_bar("ml")
    if not ml:
        st.warning("Not enough data. Need at least 30 completed/cancelled orders.")
        return

    # ── Helper ──
    def _show_base_metrics(res, icon):
        c1, c2 = st.columns(2)
        with c1:
            if res["type"] == "regression":
                rmse = res.get("rmse", "N/A")
                r2 = res.get("r2", "N/A")
                r2_color = "🟢" if r2 != "N/A" and float(r2) > 0.3 else "🟡" if r2 != "N/A" and float(r2) > 0 else "🔴"
                c1.metric(f"{icon} RMSE", rmse)
                c2.metric(f"{icon} R² {r2_color}", r2)
            elif res["type"] == "classification":
                acc = res.get("accuracy", "N/A")
                f1 = res.get("f1", "N/A")
                acc_color = "🟢" if acc != "N/A" and float(acc) > 0.8 else "🟡" if acc != "N/A" and float(acc) > 0.6 else "🔴"
                c1.metric(f"{icon} Accuracy {acc_color}", acc)
                c2.metric(f"{icon} F1-Score", f1)
        c3, c4 = st.columns(2)
        with c3:
            st.metric("Train Samples", res.get("train", "N/A"))
        with c4:
            st.metric("Test Samples", res.get("test", "N/A"))

    # ────────────────────────────────
    #  1. REGRESSION MODELS
    # ────────────────────────────────
    st.markdown("<div class='section-title'>🔮 Regression Models — نماذج التنبؤ الرقمي</div>", unsafe_allow_html=True)
    has_reg = False
    r1, r2 = st.columns(2)
    for idx, key in enumerate(["commission_predictor", "rating_predictor"]):
        res = ml.get(key)
        if not res: continue
        has_reg = True
        with (r1 if idx == 0 else r2):
            titles = {"commission_predictor": "💰 Commission Predictor — توقع العمولة",
                      "rating_predictor": "⭐ Rating Predictor — توقع التقييم"}
            targets = {"commission_predictor": "commission",
                       "rating_predictor": "rating (1-5)"}
            st.markdown(f"**{titles[key]}**<br><small>Target: {targets[key]}</small>", unsafe_allow_html=True)
            st.caption(f"Train: {res['train']} | Test: {res['test']}")
            _show_base_metrics(res, "🔮")
            fi = res.get("fi", {})
            if fi:
                dfi = pd.DataFrame(list(fi.items()), columns=["Feature","Importance"]
                                   ).sort_values("Importance", ascending=True)
                fig = px.bar(dfi, x="Importance", y="Feature", orientation="h",
                             color="Importance", color_continuous_scale="blues",
                             title="Feature Importance")
                fig.update_layout(height=200, margin=dict(l=10,r=10,t=30,b=10))
                wrap_chart(fig)
    if not has_reg:
        st.info("ℹ️ Regression models need 30+ samples with numeric features.")

    # ────────────────────────────────
    #  2. CLASSIFICATION MODELS
    # ────────────────────────────────
    st.markdown("<div class='section-title'>✅ Classification Models — نماذج التصنيف</div>", unsafe_allow_html=True)
    has_clf = False
    c1, c2 = st.columns(2)
    for idx, key in enumerate(["completion_classifier", "churn_predictor"]):
        res = ml.get(key)
        if not res: continue
        has_clf = True
        with (c1 if idx == 0 else c2):
            titles = {"completion_classifier": "📦 Order Completion — توقع إتمام الطلب",
                      "churn_predictor": "👤 Client Loyalty — توقع ولاء العميل"}
            st.markdown(f"**{titles[key]}**<br><small>Target: {res['target']}</small>", unsafe_allow_html=True)
            st.caption(f"Train: {res['train']} | Test: {res['test']}")
            _show_base_metrics(res, "✅")
            fi = res.get("fi", {})
            if fi:
                dfi = pd.DataFrame(list(fi.items()), columns=["Feature","Importance"]
                                   ).sort_values("Importance", ascending=True)
                fig = px.bar(dfi, x="Importance", y="Feature", orientation="h",
                             color="Importance", color_continuous_scale="greens",
                             title="Feature Importance")
                fig.update_layout(height=200, margin=dict(l=10,r=10,t=30,b=10))
                wrap_chart(fig)
    if not has_clf:
        st.info("ℹ️ Classification models need 30+ samples with both classes.")

    # ────────────────────────────────
    #  3. WORKER RECOMMENDER
    # ────────────────────────────────
    wr = ml.get("worker_recommender")
    if wr:
        st.markdown("<div class='section-title'>🏆 Worker Recommender — أفضل العمال لكل خدمة</div>", unsafe_allow_html=True)
        weights_str = ", ".join(f"{k.replace('_',' ').title()}: {v:.0%}" for k,v in wr['weights'].items())
        st.caption(f"Scored {wr['workers_scored']} workers across {wr['professions']} professions • "
                   f"Weights: {weights_str}")
        recs = wr.get("records", [])
        if recs:
            df_rec = pd.DataFrame(recs)
            # Show profession filter
            profs = sorted(df_rec["profession"].unique())
            sel_prof = st.selectbox("Filter by profession — اختر المهنة", ["All"] + profs)
            if sel_prof != "All":
                df_rec = df_rec[df_rec["profession"] == sel_prof]
            fig = px.bar(df_rec.sort_values("score"), x="score", y="name" if "name" in df_rec.columns else "username",
                         orientation="h", color="score", color_continuous_scale="greens",
                         text="score", title="Top Workers by Score")
            fig.update_traces(texttemplate="%{text}", textposition="outside")
            fig.update_layout(height=400 if sel_prof == "All" else 300)
            wrap_chart(fig)
            with st.expander("📋 Full Ranking Table — جدول الترتيب الكامل"):
                st.dataframe(df_rec.drop(columns=[c for c in ["_score"] if c in df_rec.columns]),
                             use_container_width=True, hide_index=True)

    # ────────────────────────────────
    #  4. WORKER CLUSTERING
    # ────────────────────────────────
    wc = ml.get("worker_clustering")
    if wc:
        st.markdown("<div class='section-title'>📊 Worker Clustering — تصنيف العمال حسب الأداء</div>", unsafe_allow_html=True)
        st.caption(f"{wc['n_clusters']} clusters • {wc['workers']} workers • Features: {', '.join(wc.get('features',[]))}")
        cents = wc.get("centroids", [])
        if cents:
            df_cent = pd.DataFrame(cents)
            feat_cols = wc.get("features", [])
            id_vars = [c for c in df_cent.columns if c in ["cluster","label"]]
            if feat_cols:
                df_melt = df_cent.melt(id_vars=id_vars, value_vars=feat_cols,
                                       var_name="Feature", value_name="Value")
                fig = px.bar(df_melt, x="Feature", y="Value", color="label",
                             barmode="group", title="Cluster Centroids — مراكز المجموعات",
                             color_discrete_sequence=px.colors.qualitative.Set2)
                wrap_chart(fig)
        assigns = wc.get("assignments", [])
        if assigns:
            df_assign = pd.DataFrame(assigns)
            col_order = ["label"] + [c for c in df_assign.columns if c != "label"]
            df_assign = df_assign[[c for c in col_order if c in df_assign.columns]]
            with st.expander("📋 View Assignments — عرض التصنيفات"):
                st.dataframe(df_assign, use_container_width=True, hide_index=True)


# ──────────────────────────────────────
#  PAGE 6 — CORRELATIONS
# ──────────────────────────────────────

def page_corr():
    corr = st.session_state.get("correlations",{})
    dfs = st.session_state.dataframes
    page_header("🔗 Correlations", "الارتباطات والعلاقات بين المتغيرات")
    if not corr:
        st.info("Not enough numeric data for correlations.")
        return
    for k, v in corr.items():
        if not v:
            continue
        icon = "🔗"
        st.markdown(f"<div class='section-title'>{icon} {k.replace('_',' ').title()}</div>",
                    unsafe_allow_html=True)
        pairs = sorted(v.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
        df = pd.DataFrame(pairs, columns=["Pair","Correlation"])
        fig = px.bar(df, x="Correlation", y="Pair", orientation="h",
                     title="Top Correlations", color="Correlation",
                     color_continuous_scale="RdBu_r", range_color=[-1,1])
        wrap_chart(fig)
    o = dfs.get("orders")
    if o is not None and len(o) > 1:
        n = o.select_dtypes(include=[np.number])
        if len(n.columns) > 1:
            c = n.corr()
            fig = go.Figure(data=go.Heatmap(
                z=c.values, x=c.columns, y=c.columns,
                colorscale="RdBu_r", zmin=-1, zmax=1,
                text=c.round(2).values, texttemplate="%{text}",
                textfont=dict(size=9, color=_theme_colors()["title"])
            ))
            fig.update_layout(
                title="Correlation Heatmap", height=600,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=_theme_colors()["font"], size=10),
            )
            wrap_chart(fig)


# ──────────────────────────────────────
#  SIDEBAR NAVIGATION
# ──────────────────────────────────────

def sidebar_nav():
    with st.sidebar:
        dark = st.session_state.theme == "dark"
        import base64
        logo_path = r"C:\Users\ALAQSA\AppData\Local\Temp\opencode\service_analytics\logo app .jpeg"
        with open(logo_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        sub_color = "rgba(255,255,255,0.3)" if dark else "rgba(0,0,0,0.4)"
        st.markdown(f"""
        <div style="text-align:center; padding:8px 0 16px 0;">
            <img src="data:image/jpeg;base64,{b64}" style="height:60px; width:auto; margin-bottom:8px;">
            <div style="font-weight:700; font-size:1.1rem; background:linear-gradient(135deg,#2ecc71,#3498db); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;">Service Analytics</div>
            <div style="font-size:0.7rem; color:{sub_color}; letter-spacing:0.5px;">
                Intelligent Dashboard
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        idx = 0
        for i, (label, key) in enumerate(PAGES):
            if st.sidebar.button(label, key=f"nav_{key}"):
                st.session_state.page = key
                st.rerun()
            if st.session_state.get("page") == key:
                idx = i
        st.divider()
        if st.button("📤 Upload New Data — رفع بيانات جديدة", key="upload_btn"):
            st.session_state.loaded = False
            st.session_state._user_chose_upload = True
            st.session_state.page = "exec"
            st.rerun()
        theme_label = "☀️ Light Mode" if st.session_state.theme == "dark" else "🌙 Dark Mode"
        if st.sidebar.button(theme_label, key="theme_btn"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()
        cached = _load_cache()
        if cached:
            st.caption(f"💾 Saved — المحفوظات متاحة · {len(cached.get('dataframes',{}))} tables")


# ──────────────────────────────────────
#  MAIN
# ──────────────────────────────────────

if not st.session_state.loaded:
    cached = _load_cache()
    if cached and not st.session_state.get("_user_chose_upload"):
        for k, v in cached.items():
            if k == "dataframes":
                st.session_state.dataframes = v
            else:
                st.session_state[k] = v
        st.session_state.loaded = True
        st.session_state._user_chose_upload = False
        st.rerun()
    if cached:
        sub_color = "rgba(255,255,255,0.4)" if st.session_state.theme == "dark" else "rgba(0,0,0,0.4)"
        st.markdown(f"""
        <div style="text-align:center; margin:12px 0;">
            <p style="color:{sub_color}; font-size:0.85rem;">💾 You have saved data — لديك بيانات محفوظة من جلسة سابقة</p>
        </div>
        """, unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if st.button("📂 Resume Last Session — استئناف الجلسة", use_container_width=True, type="primary"):
                for k, v in cached.items():
                    if k == "dataframes":
                        st.session_state.dataframes = v
                    else:
                        st.session_state[k] = v
                st.session_state.loaded = True
                st.session_state._user_chose_upload = False
                st.rerun()
        st.divider()
    st.session_state.page = "exec"
    upload_page()
    if st.session_state.loaded:
        st.rerun()
else:
    if "page" not in st.session_state:
        st.session_state.page = "exec"
    sidebar_nav()
    page = st.session_state.page
    if page == "exec":
        page_exec()
    elif page == "workforce":
        page_workforce()
    elif page == "customer":
        page_customer()
    elif page == "financial":
        page_financial()
    elif page == "ml":
        page_ml()
    elif page == "corr":
        page_corr()
