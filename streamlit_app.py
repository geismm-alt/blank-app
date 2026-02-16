import streamlit as st
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import inspect

# MUST be the first Streamlit command (before any st.* calls or decorators)
st.set_page_config(
    page_title="Black-Scholes • Price, Greeks, Heatmaps, Odds",
    layout="wide",
)

# ============================================================
# THEME (Plotly charts)
# ============================================================
BG = "black"
FONT = "white"  # set to "black" for black-on-black (invisible)

# ============================================================
# Normal CDF / PDF (no SciPy)
# ============================================================
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_pdf(x: float) -> float:
    return (1.0 / (math.sqrt(2.0 * math.pi))) * math.exp(-0.5 * x * x)

# ============================================================
# Black–Scholes core
# ============================================================
def d1_d2(S, K, T, r, q, sigma):
    eps = 1e-12
    S = max(float(S), eps)
    K = max(float(K), eps)
    T = max(float(T), eps)
    sigma = max(float(sigma), eps)

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2

def bs_price_and_greeks(S, K, T, r, q, sigma, option_type: str):
    """
    Returns: price, delta, gamma, vega, theta, rho
      - theta is per YEAR
      - vega is per 1.00 change in vol (per 1% is vega*0.01)
    """
    S = float(S); K = float(K); T = float(T); r = float(r); q = float(q); sigma = float(sigma)

    if T <= 0:
        if option_type == "Call":
            price = max(S - K, 0.0)
            delta = 1.0 if S > K else 0.0
        else:
            price = max(K - S, 0.0)
            delta = -1.0 if S < K else 0.0
        return price, delta, 0.0, 0.0, 0.0, 0.0

    d1, d2 = d1_d2(S, K, T, r, q, sigma)

    Nd1  = norm_cdf(d1)
    Nd2  = norm_cdf(d2)
    Nmd1 = norm_cdf(-d1)
    Nmd2 = norm_cdf(-d2)
    pdf1 = norm_pdf(d1)

    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    if option_type == "Call":
        price = S * disc_q * Nd1 - K * disc_r * Nd2
        delta = disc_q * Nd1
        rho   = K * T * disc_r * Nd2
        theta = (
            -(S * disc_q * pdf1 * sigma) / (2.0 * math.sqrt(T))
            - r * K * disc_r * Nd2
            + q * S * disc_q * Nd1
        )
    else:
        price = K * disc_r * Nmd2 - S * disc_q * Nmd1
        delta = disc_q * (Nd1 - 1.0)
        rho   = -K * T * disc_r * Nmd2
        theta = (
            -(S * disc_q * pdf1 * sigma) / (2.0 * math.sqrt(T))
            + r * K * disc_r * Nmd2
            - q * S * disc_q * Nmd1
        )

    gamma = (disc_q * pdf1) / (S * sigma * math.sqrt(T))
    vega  = S * disc_q * pdf1 * math.sqrt(T)

    return price, delta, gamma, vega, theta, rho

# ============================================================
# Root finding (bisection) for implied vol
# ============================================================
def bracket_root(func, lo, hi, steps=60):
    xs = np.linspace(lo, hi, steps)
    fs = [func(float(x)) for x in xs]
    for i in range(len(xs) - 1):
        if fs[i] == 0:
            return float(xs[i]), float(xs[i])
        if fs[i] * fs[i + 1] < 0:
            return float(xs[i]), float(xs[i + 1])
    return None

def bisect(func, a, b, tol=1e-7, max_iter=150):
    fa = func(a)
    fb = func(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        raise ValueError("Bisection requires a sign change on [a,b].")

    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = func(m)
        if abs(fm) < tol or (b - a) < tol:
            return m
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)

# ============================================================
# Plotly styling helper (black background)
# ============================================================
def apply_theme(fig):
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=FONT),
    )
    fig.update_xaxes(title_font=dict(color=FONT), tickfont=dict(color=FONT), color=FONT)
    fig.update_yaxes(title_font=dict(color=FONT), tickfont=dict(color=FONT), color=FONT)

    try:
        fig.update_coloraxes(
            colorbar=dict(
                title=dict(font=dict(color=FONT)),
                tickfont=dict(color=FONT),
                bgcolor=BG,
                outlinecolor=BG,
            )
        )
    except Exception:
        pass

    return fig

# ============================================================
# Heatmap helpers
# ============================================================
def _fmt_spot(x: float) -> str:
    return f"{x:.1f}" if abs(x - round(x)) < 1e-9 else f"{x:.2f}"

@st.cache_data(show_spinner=False)
def compute_call_put_heatmaps(K, T, r, q, min_spot, max_spot, n_spot, min_vol, max_vol, n_vol):
    spot_vals = np.linspace(float(min_spot), float(max_spot), int(n_spot))
    vol_vals  = np.linspace(float(min_vol),  float(max_vol),  int(n_vol))

    callZ = np.zeros((len(vol_vals), len(spot_vals)), dtype=float)
    putZ  = np.zeros_like(callZ)

    for i, vol in enumerate(vol_vals):
        for j, s_ in enumerate(spot_vals):
            callZ[i, j] = bs_price_and_greeks(float(s_), float(K), float(T), float(r), float(q), float(vol), "Call")[0]
            putZ[i, j]  = bs_price_and_greeks(float(s_), float(K), float(T), float(r), float(q), float(vol), "Put")[0]

    return spot_vals, vol_vals, callZ, putZ

def make_price_heatmap(Z, spot_vals, vol_vals, title, colorscale="Viridis"):
    x_labels = [_fmt_spot(x) for x in spot_vals]
    y_labels = [f"{v:.2f}" for v in vol_vals]

    # Compatibility: some Plotly versions don't support text_auto
    try:
        if "text_auto" in inspect.signature(px.imshow).parameters:
            fig = px.imshow(
                Z,
                x=x_labels,
                y=y_labels,
                text_auto=".2f",
                aspect="auto",
                origin="upper",
                color_continuous_scale=colorscale,
                labels={"x": "Spot", "y": "Volatility", "color": "Price"},
            )
            fig.update_traces(textfont_size=11)
        else:
            fig = px.imshow(
                Z,
                x=x_labels,
                y=y_labels,
                aspect="auto",
                origin="upper",
                color_continuous_scale=colorscale,
                labels={"x": "Spot", "y": "Volatility", "color": "Price"},
            )
            fig.update_traces(text=np.round(Z, 2), texttemplate="%{text}", textfont_size=11)
    except Exception:
        # Fallback if px.imshow signature differs
        fig = go.Figure(
            data=go.Heatmap(
                z=Z,
                x=x_labels,
                y=y_labels,
                colorscale=colorscale,
                colorbar=dict(title="Price"),
                text=np.round(Z, 2),
                texttemplate="%{text}",
            )
        )

    fig.update_layout(title=title, title_x=0.5, margin=dict(l=40, r=20, t=55, b=40))
    fig.update_xaxes(title="Spot Price")
    fig.update_yaxes(title="Volatility")
    return apply_theme(fig)

def make_expensive_heatmap(Z, spot_vals, vol_vals, title, cmin=None, cmax=None, colorbar_title="Price"):
    expensive_scale = [
        (0.00, "rgb(16, 185, 129)"),  # green
        (0.50, "rgb(245, 158, 11)"),  # amber
        (1.00, "rgb(239, 68, 68)"),   # red
    ]
    fig = make_price_heatmap(Z, spot_vals, vol_vals, title, colorscale=expensive_scale)

    if cmin is not None and cmax is not None:
        fig.update_coloraxes(cmin=float(cmin), cmax=float(cmax), colorbar_title=colorbar_title)
    else:
        fig.update_coloraxes(colorbar_title=colorbar_title)

    return fig

# ============================================================
# Bell curve + odds helpers
# ============================================================
import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Normal CDF (no SciPy) ---
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _lognormal_pdf(ST: np.ndarray, S0: float, T: float, r: float, q: float, sigma: float) -> np.ndarray:
    """
    Risk-neutral Black–Scholes terminal distribution:
      ln(S_T) ~ Normal(m, s^2)
      m = ln(S0) + (r-q-0.5*sigma^2)T
      s = sigma*sqrt(T)
    Returns PDF over ST.
    """
    eps = 1e-12
    ST = np.maximum(ST, eps)
    s = max(float(sigma) * math.sqrt(max(float(T), eps)), eps)
    m = math.log(max(float(S0), eps)) + (float(r) - float(q) - 0.5 * float(sigma) ** 2) * float(T)

    z = np.log(ST)
    pdf = (1.0 / (ST * s * math.sqrt(2.0 * math.pi))) * np.exp(-((z - m) ** 2) / (2.0 * s ** 2))
    return pdf

def prob_ST_greater_than_x(S0: float, x: float, T: float, r: float, q: float, sigma: float) -> float:
    """Risk-neutral P(S_T > x) = N(d2(x))."""
    eps = 1e-12
    if T <= 0 or sigma <= 0 or x <= 0 or S0 <= 0:
        return 1.0 if S0 > x else 0.0
    d2x = (math.log(S0 / x) + (r - q - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return float(norm_cdf(d2x))

def prob_ST_less_than_x(S0: float, x: float, T: float, r: float, q: float, sigma: float) -> float:
    return 1.0 - prob_ST_greater_than_x(S0, x, T, r, q, sigma)

def _pnl_at_expiry_grid(ST: np.ndarray, K: float, premium: float, option_type: str, contract_mult: float = 100.0) -> np.ndarray:
    """
    P&L at expiry (in $ per contract by default):
      Call: max(ST-K,0) - premium
      Put : max(K-ST,0) - premium
    """
    K = float(K); premium = float(premium)
    if option_type == "Call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)
    pnl_per_share = payoff - premium
    return pnl_per_share * float(contract_mult)

def plot_bloomberg_style_pdf_and_pnl(
    S0: float, K: float, T: float, r: float, q: float, sigma: float,
    option_type: str,
    premium_per_share: float,
    contract_mult: float = 100.0,
    x_min: float | None = None,
    x_max: float | None = None,
    n_points: int = 900,
    tail_std: float = 4.0,
    bg: str = "black",
    font: str = "white",
):
    """
    Returns:
      fig, metrics dict
    Chart:
      - Left axis: terminal PDF (lognormal under RN measure)
      - Right axis: P&L at expiry vs terminal price
      - Vertical lines: Spot, Forward, Break-even
    """
    eps = 1e-12
    S0 = float(S0); K = float(K); T = float(T); r = float(r); q = float(q); sigma = float(sigma)
    premium = float(premium_per_share)

    # Forward under risk-neutral drift
    forward = S0 * math.exp((r - q) * max(T, 0.0))

    # Break-even
    if option_type == "Call":
        BE = K + premium
        p_profit = prob_ST_greater_than_x(S0, BE, T, r, q, sigma)
        p_finish_ITM = prob_ST_greater_than_x(S0, K, T, r, q, sigma)
        profit_side = f"P(S_T > BE) = {p_profit*100:.2f}%"
        itm_side = f"P(S_T > K) = {p_finish_ITM*100:.2f}%"
    else:
        BE = K - premium
        p_profit = prob_ST_less_than_x(S0, BE, T, r, q, sigma)
        p_finish_ITM = prob_ST_less_than_x(S0, K, T, r, q, sigma)
        profit_side = f"P(S_T < BE) = {p_profit*100:.2f}%"
        itm_side = f"P(S_T < K) = {p_finish_ITM*100:.2f}%"

    # Build x-range
    if T <= 0 or sigma <= 0:
        # Fallback range if inputs are degenerate
        xr_min = min(S0, K, BE) * 0.8
        xr_max = max(S0, K, BE) * 1.2
    else:
        # Range based on lognormal spread
        s = sigma * math.sqrt(T)
        m = math.log(max(S0, eps)) + (r - q - 0.5 * sigma**2) * T
        zmin = m - tail_std * s
        zmax = m + tail_std * s
        xr_min = math.exp(zmin)
        xr_max = math.exp(zmax)

    if x_min is not None: xr_min = float(x_min)
    if x_max is not None: xr_max = float(x_max)
    xr_min = max(xr_min, eps)
    xr_max = max(xr_max, xr_min * 1.01)

    ST = np.linspace(xr_min, xr_max, int(n_points))

    pdf = _lognormal_pdf(ST, S0, T, r, q, sigma) if (T > 0 and sigma > 0) else np.zeros_like(ST)
    pnl = _pnl_at_expiry_grid(ST, K, premium, option_type, contract_mult=contract_mult)

    pdf_max = float(np.max(pdf)) if np.any(pdf) else 1.0
    pnl_min = float(np.min(pnl))
    pnl_max = float(np.max(pnl))
    pad = 0.08 * (pnl_max - pnl_min + 1e-9)

    # --- Subplot with secondary y-axis (Bloomberg-like) ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # PDF (left axis)
    fig.add_trace(
        go.Scatter(
            x=ST, y=pdf,
            mode="lines",
            name="Probability Density (PDF)",
            line=dict(width=3, color="cyan"),
        ),
        secondary_y=False,
    )

    # P&L (right axis)
    fig.add_trace(
        go.Scatter(
            x=ST, y=pnl,
            mode="lines",
            name=f"Profit & Loss @ Expiry ({option_type})",
            line=dict(width=3, color="lime"),
        ),
        secondary_y=True,
    )

    # 0 P&L line
    fig.add_trace(
        go.Scatter(
            x=[xr_min, xr_max], y=[0, 0],
            mode="lines",
            name="P&L = 0",
            line=dict(width=1, dash="dot", color="rgba(255,255,255,0.55)"),
        ),
        secondary_y=True,
    )

    # Vertical reference lines (use PDF axis so they scale cleanly)
    def _vline(x, name, color, dash):
        fig.add_trace(
            go.Scatter(
                x=[x, x],
                y=[0, pdf_max],
                mode="lines",
                name=name,
                line=dict(width=2, color=color, dash=dash),
            ),
            secondary_y=False,
        )

    _vline(S0, "Current Spot", "white", "dot")
    _vline(forward, "Forward (RN)", "magenta", "dash")
    _vline(BE, "Break-even", "yellow", "dash")

    # Layout / styling
    fig.update_layout(
        title=(
            f"Terminal Distribution (Risk-Neutral) + P&L @ Expiry<br>"
            f"{profit_side} • {itm_side}"
        ),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(color=font),
        margin=dict(l=25, r=25, t=70, b=25),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.10)",
            borderwidth=1,
        ),
    )

    fig.update_xaxes(title_text="Terminal Price (S_T)")
    fig.update_yaxes(title_text="Probability Density", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text=f"Profit & Loss (≈ ${int(contract_mult)} shares)", secondary_y=True, range=[pnl_min - pad, pnl_max + pad])

    metrics = {
        "break_even": BE,
        "forward": forward,
        "p_profit": p_profit,
        "p_finish_itm": p_finish_ITM,
        "note_profit": profit_side,
        "note_itm": itm_side,
    }

    return fig, metrics


# ============================================================
# CSS / styling
# ============================================================
st.markdown(
    """
    <style>
      .stApp { background: radial-gradient(1200px 800px at 30% 10%, #111827 0%, #0b1220 45%, #070b14 100%); }
      [data-testid="stSidebar"] { background: #0b1220; border-right: 1px solid rgba(255,255,255,0.06); }
      .block-container { padding-top: 1.2rem; }
      h1, h2, h3, p, label, span { color: rgba(255,255,255,0.92) !important; }
      .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 14px 16px;
      }
      .muted { color: rgba(255,255,255,0.65) !important; }
      div[data-testid="stPlotlyChart"] > div {
        border-radius: 14px;
        padding: 10px;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
      }
      .price-row { display:flex; gap: 0.9rem; margin-top: 0.5rem; margin-bottom: 0.6rem; }
      .price-card {
        flex: 1;
        padding: 0.95rem 1.05rem;
        border-radius: 16px;
        color: white;
        border: 1px solid rgba(255,255,255,0.14);
      }
      .price-card.call {
        background: rgba(16, 185, 129, 0.22);
        border-color: rgba(16, 185, 129, 0.55);
      }
      .price-card.put  {
        background: rgba(239, 68, 68, 0.22);
        border-color: rgba(239, 68, 68, 0.55);
      }
      .price-label { font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.25rem; }
      .price-value { font-size: 2.2rem; font-weight: 750; line-height: 1.05; }
      .price-sub   { font-size: 0.9rem; opacity: 0.85; margin-top: 0.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Sidebar (SINGLE definition to avoid DuplicateWidgetID)
# ============================================================
st.sidebar.title("Black-Scholes")
st.sidebar.caption("Pricing + Greeks + heatmaps + odds")

option_type = st.sidebar.selectbox("Option Type", ["Call", "Put"], index=0, key="opt_type")

st.sidebar.subheader("Inputs")
S = st.sidebar.number_input("Spot Price (S)", value=1790.00, step=1.0, format="%.2f", key="S")
K = st.sidebar.number_input("Strike (K)", value=1850.00, step=1.0, format="%.2f", key="K")

time_mode = st.sidebar.radio("Time Input", ["Days", "Years"], horizontal=True, index=0, key="time_mode")
if time_mode == "Days":
    days = st.sidebar.number_input("Time to Expiration (days)", value=29, step=1, key="days")
    T = float(days) / 365.0
else:
    T = st.sidebar.number_input("Time to Expiration (years)", value=0.08, step=0.01, format="%.4f", key="T_years")

sigma = st.sidebar.number_input("Volatility (σ)", value=0.25, step=0.01, format="%.4f", key="sigma")
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.04, step=0.005, format="%.4f", key="r")
q = st.sidebar.number_input("Dividend Yield (q)", value=0.00, step=0.005, format="%.4f", key="q")

st.sidebar.divider()
with st.sidebar.expander("Chart settings", expanded=True):
    spot_span_pct = st.slider("Spot range ± (%)", 1, 50, 15, key="spot_span_pct")
    n_points = st.slider("Sensitivity points", 25, 400, 120, key="n_points")

# Odds input (defined ONCE)
default_premium = bs_price_and_greeks(S, K, T, r, q, sigma, option_type)[0]
with st.sidebar.expander("Odds settings", expanded=True):
    premium_per_share = st.number_input(
        "Market premium (per share)",
        value=float(default_premium),
        step=0.1,
        format="%.4f",
        key="premium_per_share",
    )

# ============================================================
# Main UI
# ============================================================
st.title("Black-Scholes Option Calculator")
st.markdown(
    "<div class='card'><span class='muted'>Enter inputs in the left sidebar. "
    "This app returns call/put prices, Greeks, implied-vol goal seek, heatmaps, and a terminal price distribution.</span></div>",
    unsafe_allow_html=True,
)

# Prices
call_price = bs_price_and_greeks(S, K, T, r, q, sigma, "Call")[0]
put_price  = bs_price_and_greeks(S, K, T, r, q, sigma, "Put")[0]

st.markdown(
    f"""
<div class="price-row">
  <div class="price-card call">
    <div class="price-label">Call price</div>
    <div class="price-value">{call_price:,.4f}</div>
    <div class="price-sub">Per contract (×100): {call_price*100:,.2f}</div>
  </div>
  <div class="price-card put">
    <div class="price-label">Put price</div>
    <div class="price-value">{put_price:,.4f}</div>
    <div class="price-sub">Per contract (×100): {put_price*100:,.2f}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Greeks for selected option_type
price, delta, gamma, vega, theta, rho = bs_price_and_greeks(S, K, T, r, q, sigma, option_type)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"{option_type} Price", f"{price:,.4f}")
c2.metric("Delta", f"{delta:,.6f}")
c3.metric("Gamma", f"{gamma:,.6f}")
c4.metric("Vega (per 1.00 σ)", f"{vega:,.4f}")
c5.metric("Theta / year", f"{theta:,.4f}")

tabs = st.tabs(["Price & Greeks", "Goal Seek (Implied σ)", "Sensitivity + Heatmaps", "Bell Curve (Odds)"])

# ============================================================
# Tab 1: Price & Greeks table
# ============================================================
with tabs[0]:
    left, right = st.columns([1.2, 1.0], gap="large")

    with left:
        st.subheader("Greeks (summary)")
        df = pd.DataFrame(
            {
                "Metric": ["Price", "Delta", "Gamma", "Vega", "Theta", "Rho"],
                "Value": [price, delta, gamma, vega, theta, rho],
                "Notes": [
                    "Option value",
                    "dPrice/dS",
                    "d²Price/dS²",
                    "dPrice/dσ (per 1.00)",
                    "per year",
                    "dPrice/dr",
                ],
            }
        )
        st.dataframe(df.style.format({"Value": "{:,.6f}"}), use_container_width=True, hide_index=True)

        st.markdown(
            "<div class='card'><span class='muted'>Conversions:</span><br>"
            f"• Vega per 1% σ ≈ <b>{vega*0.01:,.4f}</b><br>"
            f"• Theta per day ≈ <b>{theta/365.0:,.6f}</b></div>",
            unsafe_allow_html=True,
        )

    with right:
        st.subheader("Inputs (recap)")
        st.markdown(
            f"""
            <div class="card">
              <div><span class="muted">Type</span>: <b>{option_type}</b></div>
              <div><span class="muted">S</span>: <b>{S:,.2f}</b></div>
              <div><span class="muted">K</span>: <b>{K:,.2f}</b></div>
              <div><span class="muted">T</span>: <b>{T:.6f}</b> years</div>
              <div><span class="muted">σ</span>: <b>{sigma:.6f}</b></div>
              <div><span class="muted">r</span>: <b>{r:.6f}</b></div>
              <div><span class="muted">q</span>: <b>{q:.6f}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# Tab 2: Goal seek implied vol
# ============================================================
with tabs[1]:
    st.subheader("Goal Seek σ")
    st.caption("Find σ that matches a target market price (bisection).")

    sigma_lo = st.number_input("σ lower bound", value=0.0001, step=0.0001, format="%.6f", key="sigma_lo")
    sigma_hi = st.number_input("σ upper bound", value=5.0000, step=0.1, format="%.6f", key="sigma_hi")
    target_price = st.number_input("Target Option Price", value=float(price), step=0.1, format="%.6f", key="target_price")

    def f(sig):
        p, *_ = bs_price_and_greeks(S, K, T, r, q, sig, option_type)
        return p - target_price

    if st.button("Compute Implied σ", type="primary"):
        try:
            br = bracket_root(f, sigma_lo, sigma_hi)
            if br is None:
                st.error("Could not bracket a solution in the given σ range. Expand the bounds.")
            else:
                a, b = br
                implied = bisect(f, a, b)
                p2, d2_, g2_, v2_, t2_, r2_ = bs_price_and_greeks(S, K, T, r, q, implied, option_type)
                st.success(f"Implied σ ≈ {implied:.6f}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Price @ implied σ", f"{p2:,.6f}")
                m2.metric("Delta", f"{d2_:,.6f}")
                m3.metric("Vega (per 1% σ)", f"{v2_*0.01:,.4f}")
        except Exception as e:
            st.error(f"Error: {e}")

# ============================================================
# Tab 3: Sensitivity + Heatmaps
# ============================================================
with tabs[2]:
    st.subheader("Sensitivity vs Spot (holding other inputs constant)")

    span = max(1.0, float(spot_span_pct)) / 100.0
    S_min = max(0.01, S * (1.0 - span))
    S_max = S * (1.0 + span)
    grid_S = np.linspace(S_min, S_max, int(n_points))

    prices_list, deltas_list, gammas_list, vegas_list, thetas_list, rhos_list = [], [], [], [], [], []
    for s_ in grid_S:
        p, d, g, v, t, rh = bs_price_and_greeks(float(s_), K, T, r, q, sigma, option_type)
        prices_list.append(p)
        deltas_list.append(d)
        gammas_list.append(g)
        vegas_list.append(v * 0.01)      # per 1% vol
        thetas_list.append(t / 365.0)    # per day
        rhos_list.append(rh)

    chart_df = pd.DataFrame(
        {
            "Spot": grid_S,
            "Price": prices_list,
            "Delta": deltas_list,
            "Gamma": gammas_list,
            "Vega (per 1% σ)": vegas_list,
            "Theta (per day)": thetas_list,
            "Rho": rhos_list,
        }
    )

    y_choice = st.selectbox(
        "Plot",
        ["Price", "Delta", "Gamma", "Vega (per 1% σ)", "Theta (per day)", "Rho"],
        index=0,
        key="y_choice",
    )

    fig_line = px.line(chart_df, x="Spot", y=y_choice)
    fig_line.update_layout(margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Spot (S)")
    st.plotly_chart(apply_theme(fig_line), use_container_width=True)

    with st.expander("Download sensitivity table"):
        st.dataframe(chart_df, use_container_width=True)
        st.download_button(
            "Download CSV",
            data=chart_df.to_csv(index=False).encode("utf-8"),
            file_name="bs_sensitivity.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.subheader("Price Heatmaps (Volatility vs Spot)")

    with st.expander("Heatmap settings", expanded=False):
        heat_spot_min = st.number_input("Spot min", value=float(S * 0.80), format="%.2f", key="heat_spot_min")
        heat_spot_max = st.number_input("Spot max", value=float(S * 1.20), format="%.2f", key="heat_spot_max")
        heat_spot_points = st.number_input("Spot points", value=10, min_value=5, max_value=30, step=1, key="heat_spot_points")

        heat_vol_min = st.number_input("Vol min", value=float(max(0.01, sigma * 0.50)), format="%.4f", key="heat_vol_min")
        heat_vol_max = st.number_input("Vol max", value=float(max(0.02, sigma * 1.50)), format="%.4f", key="heat_vol_max")
        heat_vol_points = st.number_input("Vol points", value=10, min_value=5, max_value=30, step=1, key="heat_vol_points")

    spot_vals, vol_vals, callZ, putZ = compute_call_put_heatmaps(
        K=K, T=T, r=r, q=q,
        min_spot=heat_spot_min, max_spot=heat_spot_max, n_spot=int(heat_spot_points),
        min_vol=heat_vol_min, max_vol=heat_vol_max, n_vol=int(heat_vol_points),
    )

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Call Price Heatmap")
        st.plotly_chart(make_price_heatmap(callZ, spot_vals, vol_vals, "CALL price"), use_container_width=True)
    with colB:
        st.subheader("Put Price Heatmap")
        st.plotly_chart(make_price_heatmap(putZ, spot_vals, vol_vals, "PUT price"), use_container_width=True)

    st.markdown("### Expensive vs Cheap (green → red)")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        show_per_contract = st.checkbox("Show per contract (×100)", value=True, key="show_per_contract")
    with c2:
        contract_mult = st.number_input("Contract multiplier", value=100, min_value=1, step=1, key="contract_mult")
    with c3:
        clip_pct = st.slider("Clip color scale (percentiles)", 0, 10, 2, key="clip_pct")

    mult = float(contract_mult) if show_per_contract else 1.0
    call_cost = callZ * mult
    put_cost  = putZ * mult
    combined = np.concatenate([call_cost.ravel(), put_cost.ravel()])

    if clip_pct > 0:
        cmin = float(np.percentile(combined, clip_pct))
        cmax = float(np.percentile(combined, 100 - clip_pct))
    else:
        cmin = float(combined.min())
        cmax = float(combined.max())

    unit = "Contract Cost" if show_per_contract else "Option Price"

    colA, colB = st.columns(2)
    with colA:
        st.subheader("CALL (expensiveness)")
        st.plotly_chart(
            make_expensive_heatmap(call_cost, spot_vals, vol_vals, "CALL", cmin=cmin, cmax=cmax, colorbar_title=unit),
            use_container_width=True,
        )
    with colB:
        st.subheader("PUT (expensiveness)")
        st.plotly_chart(
            make_expensive_heatmap(put_cost, spot_vals, vol_vals, "PUT", cmin=cmin, cmax=cmax, colorbar_title=unit),
            use_container_width=True,
        )

# ============================================================
# Tab 4: Bell curve / odds
# ============================================================
with tabs[3]:
    st.subheader("Bell Curve / Bloomberg-style Odds")

    fig, m = plot_bloomberg_style_pdf_and_pnl(
        S0=float(S),
        K=float(K),
        T=float(T),
        r=float(r),
        q=float(q),
        sigma=float(sigma),
        option_type=option_type,
        premium_per_share=float(premium_per_share),
        contract_mult=100.0,          # change if needed
        # Optional: match Bloomberg-like range manually
        # x_min=900, x_max=2500,
        bg=BG,
        font=FONT,
    )
    st.plotly_chart(fig, use_container_width=True)

    a, b, c, d = st.columns(4)
    a.metric("Break-even (per share)", f"{m['break_even']:,.2f}")
    b.metric("Forward (RN)", f"{m['forward']:,.2f}")
    c.metric("P(profit at expiry)", f"{m['p_profit']*100:.2f}%")
    d.metric("P(finish ITM)", f"{m['p_finish_itm']*100:.2f}%")

    st.markdown(
        """
**How to read this (matches what you see on Bloomberg):**
- **Cyan curve = Probability Density (PDF)** of the terminal price **S_T** under Black–Scholes (risk-neutral).  
  Higher points mean “more likely outcomes”; the **area** under the curve between two prices is the probability of landing in that range.
- **Green line = Profit & Loss at expiry** for *one contract* (default ×100 shares).  
  Where it crosses **P&L = 0** is your **break-even**.
- **Break-even line (yellow)** is the price you need at expiration to have **zero** profit/loss:
  - Call: **BE = K + premium**
  - Put: **BE = K − premium**
- **P(profit at expiry)** is the risk-neutral probability that **S_T ends past break-even** (not a guarantee).

**Legend interpretation:**
- If most of the cyan area lies **left** of break-even, the market-implied odds of profit are low.
- Even if P(profit) is low, the trade can still make sense if your *view* differs (higher realized vol, catalyst, skew, etc.).
        """
    )
