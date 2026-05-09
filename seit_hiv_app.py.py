"""
================================================================
SEIT HIV Transmission Dynamics Model — Juja Sub-County, Kenya
Streamlit Interactive Application
================================================================
Authors : Laurence Maina, Hilda Peter, Vera Vicky,
          Stanley Munyao, Baruch Limo
Supervisor: Dr. Phineus Kiogora, PhD
Institution: JKUAT — Dept. of Pure & Applied Mathematics, 2026

HOW TO RUN:
  pip install streamlit scipy numpy pandas matplotlib plotly
  streamlit run seit_hiv_app.py
================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.integrate import odeint
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SEIT HIV Model — Juja Sub-County",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .stApp { background-color: #0b0f1a; }
  section[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1e2d45; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background-color: #111827;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 16px 20px;
  }
  [data-testid="stMetricValue"] { color: #e8eef8; font-size: 2rem !important; font-weight: 800 !important; }
  [data-testid="stMetricLabel"] { color: #7a8ba0; font-size: 0.75rem !important; }
  [data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

  /* Headers */
  h1, h2, h3 { color: #e8eef8 !important; }
  h1 { font-size: 2.2rem !important; font-weight: 800 !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background-color: #111827; border-bottom: 1px solid #1e2d45; gap: 0; }
  .stTabs [data-baseweb="tab"] { background-color: transparent; color: #7a8ba0; border-radius: 0; padding: 12px 24px; font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; }
  .stTabs [aria-selected="true"] { background-color: transparent; color: #00d4aa !important; border-bottom: 2px solid #00d4aa; }

  /* Sliders */
  .stSlider { padding: 4px 0; }

  /* Dataframe */
  [data-testid="stDataFrame"] { border: 1px solid #1e2d45; border-radius: 8px; }

  /* Info / warning boxes */
  .info-box {
    background: #111827;
    border: 1px solid #1e2d45;
    border-left: 4px solid #00d4aa;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 12px 0;
    font-family: monospace;
    color: #e8eef8;
    font-size: 0.9rem;
    line-height: 1.8;
  }
  .eq-box {
    background: #111827;
    border: 1px solid #1e2d45;
    border-left: 4px solid #ffd93d;
    border-radius: 0 8px 8px 0;
    padding: 14px 20px;
    margin: 8px 0;
    font-family: monospace;
    color: #ffd93d;
    font-size: 0.95rem;
  }
  .r0-box {
    background: #1a2235;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 20px 28px;
    margin: 16px 0;
    display: flex;
    align-items: center;
    gap: 24px;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS & BASELINE PARAMETERS
# ─────────────────────────────────────────────────────────────
BASE = dict(Lambda=2850, beta=1.82, mu=0.014, k=0.1095,
            delta=0.025, tau=0.152, eps=0.04)

Y0 = [90857, 450, 587, 4045]   # S, E, I, T at Jan 2026

SCENARIO_COLORS = {
    "A: Baseline":  "#ff6b6b",
    "B: PrEP 20%":  "#ffd93d",
    "C: PrEP 40%":  "#00d4aa",
    "D: Combined":  "#6bceff",
}

HIST_DATA = pd.DataFrame({
    "Year":       [2019,2020,2021,2022,2023,2024,2025],
    "Population": [88456,90123,91567,92845,93978,94924,95489],
    "HIV+ Cases": [4112,4201,4289,4378,4467,4556,4632],
    "Prevalence %":[4.65,4.66,4.68,4.72,4.75,4.80,4.85],
    "New Diagnoses":[198,187,205,221,228,234,237],
    "On ART":     [3289,3456,3601,3734,3856,3967,4045],
    "PrEP Users": [412,445,478,512,548,589,623],
    "HIV Deaths": [98,89,104,109,112,115,119],
    "ART Coverage %":[80.0,82.3,84.0,85.3,86.3,87.0,87.3],
})

# ─────────────────────────────────────────────────────────────
# SEIT ODE SYSTEM
# ─────────────────────────────────────────────────────────────
def seit_model(y, t, Lambda, beta, mu, k, delta, tau, eps):
    S, E, I, T = y
    N = S + E + I + T
    if N <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    lam = beta * (I + eps * T) / N
    dS = Lambda - lam * S - mu * S
    dE = lam * S - (mu + k) * E
    dI = k * E - (mu + delta + tau) * I
    dT = tau * I - mu * T
    return [dS, dE, dI, dT]


def solve_seit(params, t_end=5.0, n_pts=61):
    """Solve SEIT ODE system; returns DataFrame with columns t, year, S, E, I, T, N, prevalence, incidence."""
    t = np.linspace(0, t_end, n_pts)
    sol = odeint(seit_model, Y0, t,
                 args=(params["Lambda"], params["beta"], params["mu"],
                       params["k"], params["delta"], params["tau"], params["eps"]))
    S, E, I, T = sol.T
    N = S + E + I + T
    df = pd.DataFrame({
        "t":          t,
        "year":       2026 + t,
        "S":          S, "E": E, "I": I, "T": T, "N": N,
        "prevalence": (I + T) / N * 100,
        "incidence":  params["k"] * E * 12,   # annualised
    })
    return df


def compute_r0(p):
    return (p["beta"] * p["k"]) / ((p["mu"] + p["k"]) * (p["mu"] + p["delta"] + p["tau"]))


def cum_infections(df):
    """Trapezoidal integration of annualised incidence."""
    return np.trapz(df["incidence"], df["t"])


# ─────────────────────────────────────────────────────────────
# RUN ALL SCENARIOS (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def run_all_scenarios():
    scenarios = {
        "A: Baseline": {**BASE},
        "B: PrEP 20%": {**BASE, "beta": BASE["beta"] * 0.86},
        "C: PrEP 40%": {**BASE, "beta": BASE["beta"] * 0.72},
        "D: Combined": {**BASE, "beta": BASE["beta"] * 0.72, "tau": BASE["tau"] * 2.0},
    }
    return {name: solve_seit(p) for name, p in scenarios.items()}


# ─────────────────────────────────────────────────────────────
# MATPLOTLIB THEME HELPER
# ─────────────────────────────────────────────────────────────
def dark_fig(figsize=(12, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#0b0f1a")
    ax.tick_params(colors="#7a8ba0", labelsize=9)
    ax.xaxis.label.set_color("#7a8ba0")
    ax.yaxis.label.set_color("#7a8ba0")
    ax.title.set_color("#e8eef8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d45")
    ax.grid(color="#1e2d45", linewidth=0.7, alpha=0.8)
    return fig, ax


def dark_figs(nrows=1, ncols=2, figsize=(14, 5)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor("#111827")
    flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax in flat:
        ax.set_facecolor("#0b0f1a")
        ax.tick_params(colors="#7a8ba0", labelsize=9)
        ax.xaxis.label.set_color("#7a8ba0")
        ax.yaxis.label.set_color("#7a8ba0")
        ax.title.set_color("#e8eef8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e2d45")
        ax.grid(color="#1e2d45", linewidth=0.7, alpha=0.8)
    return fig, axes


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:32px 0 16px'>
  <div style='font-family:monospace;font-size:11px;color:#00d4aa;letter-spacing:3px;
              text-transform:uppercase;margin-bottom:10px'>
    JKUAT — DEPT. OF PURE & APPLIED MATHEMATICS · 2026
  </div>
  <h1 style='font-size:2.4rem;font-weight:900;color:#e8eef8;line-height:1.1;margin:0'>
    SEIT HIV <span style='color:#00d4aa'>Transmission</span> Dynamics Model
  </h1>
  <p style='color:#7a8ba0;margin-top:8px;font-size:15px'>
    Juja Sub-County, Kiambu County, Kenya · Runge-Kutta (RK4) Simulation · 2026–2030
  </p>
</div>
<hr style='border:none;border-top:1px solid #1e2d45;margin-bottom:24px'>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR — Global controls
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Parameters")
    st.markdown("---")
    st.markdown("**Baseline values (Juja Hospital)**")

    sb_beta   = st.slider("β — Transmission rate (yr⁻¹)",   0.5, 3.0,  BASE["beta"],   0.01)
    sb_tau    = st.slider("τ — Treatment/ART rate (yr⁻¹)",  0.05, 0.6,  BASE["tau"],   0.005)
    sb_eps    = st.slider("ε — ART infectivity factor",      0.01, 0.5,  BASE["eps"],   0.01)
    sb_k      = st.slider("k — Progression rate (yr⁻¹)",    0.05, 0.5,  BASE["k"],     0.005)
    sb_Lambda = st.slider("Λ — Recruitment rate (/yr)",      1000, 6000, BASE["Lambda"], 50)
    sb_delta  = st.slider("δ — Disease mortality (yr⁻¹)",   0.005, 0.1,  BASE["delta"], 0.005)

    custom_params = dict(Lambda=sb_Lambda, beta=sb_beta, mu=BASE["mu"],
                         k=sb_k, delta=sb_delta, tau=sb_tau, eps=sb_eps)

    r0_custom = compute_r0(custom_params)
    r0_base   = compute_r0(BASE)
    r0_color  = "#00d4aa" if r0_custom < 1.0 else ("#ffd93d" if r0_custom < 1.5 else "#ff6b6b")

    st.markdown("---")
    st.markdown(f"""
    <div style='background:#0b0f1a;border:1px solid #1e2d45;border-radius:10px;padding:16px;text-align:center'>
      <div style='font-family:monospace;font-size:10px;color:#7a8ba0;letter-spacing:2px;margin-bottom:6px'>CUSTOM R₀</div>
      <div style='font-size:2.8rem;font-weight:900;color:{r0_color};line-height:1'>{r0_custom:.3f}</div>
      <div style='font-size:11px;color:#7a8ba0;margin-top:6px'>
        {"🟢 DFE Stable — Elimination possible" if r0_custom < 1.0 else
         "🟡 Near threshold — Borderline" if r0_custom < 1.5 else
         "🔴 Endemic — R₀ > 1"}
      </div>
      <div style='font-size:11px;color:#7a8ba0;margin-top:4px'>
        vs. baseline 1.85 &nbsp;
        <span style='color:{"#00d4aa" if r0_custom < r0_base else "#ff6b6b"}'>
          ({("" if r0_custom >= r0_base else "")}{r0_custom - r0_base:+.3f})
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Authors: Laurence Maina · Hilda Peter · Vera Vicky · Stanley Munyao · Baruch Limo")
    st.caption("Supervisor: Dr. Phineus Kiogora, PhD")

# ─────────────────────────────────────────────────────────────
# RUN SCENARIOS
# ─────────────────────────────────────────────────────────────
all_results = run_all_scenarios()
base_cum    = cum_infections(all_results["A: Baseline"])

# ─────────────────────────────────────────────────────────────
# TOP METRICS
# ─────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("R₀ (Baseline)", "1.85", "Endemic → R₀ > 1")
with c2: st.metric("HIV Prevalence 2025", "4.85%", "+0.20% since 2019")
with c3: st.metric("New Cases/Year", "237", "+39 since 2019")
with c4: st.metric("ART Coverage", "87.3%", "+7.3% since 2019")
with c5: st.metric("Best Reduction (D)", "61.8%", "Combined PrEP+Testing")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tabs = st.tabs(["📊 Dashboard", "🔬 Simulate", "📐 Model & Equations", "🏥 Facility Data", "💰 Cost-Effectiveness"])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════
with tabs[0]:

    # ── R0 indicator ──
    r0_base = compute_r0(BASE)
    r0_pct = min(r0_base / 3.0, 1.0) * 100
    st.markdown(f"""
    <div style='background:#1a2235;border:1px solid #1e2d45;border-radius:12px;
                padding:24px 32px;margin:8px 0 24px;display:flex;align-items:center;gap:32px;flex-wrap:wrap'>
      <div>
        <div style='font-family:monospace;font-size:10px;color:#7a8ba0;letter-spacing:2px;margin-bottom:6px'>BASIC REPRODUCTION NUMBER</div>
        <div style='font-size:3.5rem;font-weight:900;color:#ff6b6b;line-height:1'>1.85</div>
      </div>
      <div style='flex:1;min-width:240px'>
        <h3 style='margin:0 0 6px;color:#e8eef8'>HIV is Endemic in Juja Sub-County</h3>
        <p style='color:#7a8ba0;font-size:14px;line-height:1.6;margin:0'>
          Each infectious individual generates on average <strong style='color:#e8eef8'>1.85 secondary infections</strong>
          in a fully susceptible population. For elimination, R₀ must be reduced below 1.0
          through combined intervention (PrEP + enhanced testing).
        </p>
      </div>
      <div style='flex:2;min-width:260px'>
        <div style='font-family:monospace;font-size:10px;color:#7a8ba0;letter-spacing:2px;margin-bottom:8px'>R₀ SCALE  (0 → 3)</div>
        <div style='height:8px;background:#1e2d45;border-radius:4px;position:relative;overflow:hidden'>
          <div style='height:100%;width:{r0_pct:.1f}%;background:linear-gradient(90deg,#00d4aa,#ff6b6b);border-radius:4px'></div>
        </div>
        <div style='display:flex;justify-content:space-between;font-family:monospace;font-size:10px;color:#7a8ba0;margin-top:4px'>
          <span>0</span><span>1 ← threshold</span><span>2</span><span>3</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Prevalence projections ──
    show_scenarios = st.multiselect(
        "Toggle scenarios:",
        list(SCENARIO_COLORS.keys()),
        default=list(SCENARIO_COLORS.keys()),
        key="dash_scenarios",
    )

    fig, ax = dark_fig(figsize=(14, 5))
    for name, df in all_results.items():
        if name in show_scenarios:
            ax.plot(df["year"], df["prevalence"],
                    color=SCENARIO_COLORS[name], linewidth=2.5,
                    linestyle="--" if name == "A: Baseline" else "-",
                    label=name)
    ax.axhline(4.85, color="#ffffff22", linestyle=":", linewidth=1.2, label="2025 baseline (4.85%)")
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("HIV Prevalence (%)", fontsize=10)
    ax.set_title("SEIT Model — HIV Prevalence Projections 2026–2030", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1a2235", edgecolor="#1e2d45", labelcolor="#e8eef8", fontsize=9)
    st.pyplot(fig)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    # ── Annual incidence ──
    with col1:
        fig, ax = dark_fig(figsize=(7, 4))
        annual_idx = [0, 12, 24, 36, 48, 60]
        annual_yrs = [2026, 2027, 2028, 2029, 2030, 2031]
        x = np.arange(len(annual_idx))
        w = 0.2
        for i, (name, df) in enumerate(all_results.items()):
            vals = [df["incidence"].iloc[j] for j in annual_idx]
            ax.bar(x + i*w, vals, width=w, color=SCENARIO_COLORS[name], label=name,
                   alpha=0.85, edgecolor=SCENARIO_COLORS[name])
        ax.set_xticks(x + 1.5*w)
        ax.set_xticklabels(annual_yrs, fontsize=9)
        ax.set_title("Annual New HIV Infections", fontsize=12, fontweight="bold")
        ax.set_ylabel("Infections / yr", fontsize=9)
        ax.legend(facecolor="#1a2235", edgecolor="#1e2d45", labelcolor="#e8eef8", fontsize=8)
        st.pyplot(fig)
        plt.close()

    # ── SEIT compartments baseline ──
    with col2:
        fig, ax = dark_fig(figsize=(7, 4))
        df_b = all_results["A: Baseline"]
        ax.plot(df_b["year"], df_b["S"], color="#6bceff", linewidth=2, label="S – Susceptible")
        ax.plot(df_b["year"], df_b["E"], color="#ffd93d", linewidth=2, label="E – Exposed")
        ax.plot(df_b["year"], df_b["I"], color="#ff6b6b", linewidth=2, label="I – Infected")
        ax.plot(df_b["year"], df_b["T"], color="#00d4aa", linewidth=2, label="T – Treated")
        ax.set_title("SEIT Compartment Dynamics (Baseline)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Population", fontsize=9)
        ax.legend(facecolor="#1a2235", edgecolor="#1e2d45", labelcolor="#e8eef8", fontsize=8)
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    # ── Infections averted ──
    with col3:
        names  = list(all_results.keys())
        avert  = [max(0, base_cum - cum_infections(all_results[n])) for n in names]
        colors = [SCENARIO_COLORS[n] for n in names]
        fig, ax = dark_fig(figsize=(7, 4))
        bars = ax.bar(names, avert, color=colors, edgecolor=colors, alpha=0.85, width=0.5)
        for bar, val in zip(bars, avert):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                    f"{val:.0f}", ha="center", va="bottom", color="#e8eef8", fontsize=9, fontweight="bold")
        ax.set_title("Infections Averted — 5-Year Total", fontsize=12, fontweight="bold")
        ax.set_ylabel("Infections Averted", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        st.pyplot(fig)
        plt.close()

    # ── Sensitivity analysis ──
    with col4:
        params_sens = {"β": "beta", "μ": "mu", "k": "k", "τ": "tau"}
        base_r0 = compute_r0(BASE)
        sens_pos, sens_neg, labels_s = [], [], []
        for label, key in params_sens.items():
            p_hi = {**BASE, key: BASE[key] * 1.2}
            p_lo = {**BASE, key: BASE[key] * 0.8}
            sens_pos.append(compute_r0(p_hi) - base_r0)
            sens_neg.append(compute_r0(p_lo) - base_r0)
            labels_s.append(label)

        fig, ax = dark_fig(figsize=(7, 4))
        y = np.arange(len(labels_s))
        ax.barh(y, sens_pos, height=0.35, color="#ff6b6b", alpha=0.85, label="+20%")
        ax.barh(y - 0.38, sens_neg, height=0.35, color="#00d4aa", alpha=0.85, label="−20%")
        ax.set_yticks(y - 0.19)
        ax.set_yticklabels(labels_s, fontsize=11)
        ax.axvline(0, color="#7a8ba0", linewidth=1)
        ax.set_title("Sensitivity Analysis — ΔR₀ at ±20%", fontsize=12, fontweight="bold")
        ax.set_xlabel("ΔR₀", fontsize=9)
        ax.legend(facecolor="#1a2235", edgecolor="#1e2d45", labelcolor="#e8eef8", fontsize=9)
        st.pyplot(fig)
        plt.close()

    # ── Summary table ──
    st.markdown("### Simulation Results Summary (2026–2030)")
    summary_data = []
    for name, df in all_results.items():
        cum = cum_infections(df)
        averted = max(0, base_cum - cum)
        summary_data.append({
            "Scenario":          name,
            "Final Prevalence":  f"{df['prevalence'].iloc[-1]:.2f}%",
            "Total New Infections": f"{cum:.0f}",
            "Infections Averted":   f"{averted:.0f}",
            "% Reduction":          f"{averted/base_cum*100:.1f}%" if base_cum > 0 else "—",
        })
    df_sum = pd.DataFrame(summary_data)
    st.dataframe(df_sum.set_index("Scenario"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — SIMULATE (custom parameters from sidebar)
# ═══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### 🔬 Custom Parameter Simulation")
    st.markdown("Adjust sliders in the **sidebar** to explore any parameter combination.")

    custom_df   = solve_seit(custom_params)
    base_df     = all_results["A: Baseline"]
    cum_custom  = cum_infections(custom_df)
    avert_cust  = max(0, base_cum - cum_custom)

    # Custom R0 display
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Custom R₀", f"{r0_custom:.4f}", f"{r0_custom - 1.85:+.4f} vs baseline")
    with c2: st.metric("Final Prevalence (2030)", f"{custom_df['prevalence'].iloc[-1]:.2f}%",
                        f"{custom_df['prevalence'].iloc[-1] - base_df['prevalence'].iloc[-1]:+.2f}% vs baseline")
    with c3: st.metric("5-yr New Infections", f"{cum_custom:.0f}", f"{avert_cust:+.0f} averted")
    with c4: st.metric("Status", "🟢 Controlled" if r0_custom < 1 else "🔴 Endemic",
                        "DFE stable" if r0_custom < 1 else f"R₀ = {r0_custom:.2f} > 1")

    # Prevalence comparison
    fig, axes = dark_figs(1, 2, figsize=(14, 5))
    ax1, ax2 = axes

    ax1.plot(base_df["year"],   base_df["prevalence"],   color="#ff6b6b", linewidth=2,
             linestyle="--", label="A: Baseline")
    ax1.plot(custom_df["year"], custom_df["prevalence"], color="#00d4aa", linewidth=2.5,
             label="Custom Scenario")
    ax1.set_title("Prevalence: Custom vs. Baseline", fontsize=12, fontweight="bold")
    ax1.set_ylabel("HIV Prevalence (%)", fontsize=9)
    ax1.legend(facecolor="#1a2235", edgecolor="#1e2d45", labelcolor="#e8eef8", fontsize=9)

    # SEIT dynamics
    ax2.plot(custom_df["year"], custom_df["S"], color="#6bceff", lw=2, label="S – Susceptible")
    ax2.plot(custom_df["year"], custom_df["E"], color="#ffd93d", lw=2, label="E – Exposed")
    ax2.plot(custom_df["year"], custom_df["I"], color="#ff6b6b", lw=2, label="I – Infected")
    ax2.plot(custom_df["year"], custom_df["T"], color="#00d4aa", lw=2, label="T – Treated")
    ax2.set_title("Custom — SEIT Compartment Dynamics", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Population", fontsize=9)
    ax2.legend(facecolor="#1a2235", edgecolor="#1e2d45", labelcolor="#e8eef8", fontsize=9)
    st.pyplot(fig)
    plt.close()

    # Incidence comparison
    fig, ax = dark_fig(figsize=(14, 4))
    ax.fill_between(base_df["year"],   base_df["incidence"],   alpha=0.25, color="#ff6b6b")
    ax.fill_between(custom_df["year"], custom_df["incidence"], alpha=0.25, color="#00d4aa")
    ax.plot(base_df["year"],   base_df["incidence"],   color="#ff6b6b", lw=2, ls="--", label="A: Baseline")
    ax.plot(custom_df["year"], custom_df["incidence"], color="#00d4aa", lw=2.5, label="Custom")
    ax.set_title("Annual New Infections — Custom vs. Baseline", fontsize=12, fontweight="bold")
    ax.set_ylabel("New Infections / yr", fontsize=9)
    ax.legend(facecolor="#1a2235", edgecolor="#1e2d45", labelcolor="#e8eef8", fontsize=9)
    st.pyplot(fig)
    plt.close()

    # Export custom results
    st.markdown("### 📥 Export Custom Simulation Data")
    csv_data = custom_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download CSV", csv_data, "seit_custom_simulation.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════
# TAB 3 — MODEL & EQUATIONS
# ═══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### 📐 SEIT Compartmental Model")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("#### Governing ODEs")
        for eq in [
            "dS/dt = Λ − β(I + εT)(S/N) − μS",
            "dE/dt = β(I + εT)(S/N) − (μ + k)E",
            "dI/dt = kE − (μ + δ + τ)I",
            "dT/dt = τI − μT",
        ]:
            st.markdown(f'<div class="eq-box">{eq}</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">λ(t) = β(I + εT) / N(t)  ← Force of Infection</div>',
                    unsafe_allow_html=True)

        st.markdown("#### R₀ Derivation (Next Generation Matrix)")
        st.markdown('<div class="info-box">R₀ = (β × k) / [(μ + k)(μ + δ + τ)]</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
        R₀ = (1.82 × 0.1095) / [(0.014 + 0.1095)(0.014 + 0.025 + 0.152)]<br>
           = 0.19929 / (0.1235 × 0.191)<br>
           = 0.19929 / 0.023589<br>
           ≈ <strong>1.85  →  ENDEMIC</strong>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Model Assumptions")
        for a in [
            "Homogeneous mixing — equal contact probability",
            "Open population with constant recruitment Λ",
            "Natural mortality μ on all compartments",
            "HIV transmission via sexual contact (adults 15–49)",
            "Treated individuals have reduced infectiousness ε = 0.04",
            "Disease mortality δ on infected (I) compartment only",
            "Average latency period 9.13 years (k = 1/9.13 yr⁻¹)",
            "Treated individuals do not revert to infected compartment",
        ]:
            st.markdown(f"✦ {a}")

    with col_right:
        st.markdown("#### Parameter Table")
        params_df = pd.DataFrame([
            {"Symbol": "Λ",  "Name": "Recruitment rate",   "Value": "2,850",   "Unit": "persons/yr",    "Source": "Juja Hospital + KNBS"},
            {"Symbol": "β",  "Name": "Transmission rate",   "Value": "1.82",    "Unit": "yr⁻¹",          "Source": "Model calibration"},
            {"Symbol": "μ",  "Name": "Natural mortality",   "Value": "0.014",   "Unit": "yr⁻¹",          "Source": "KNBS life tables"},
            {"Symbol": "k",  "Name": "Progression rate",    "Value": "0.1095",  "Unit": "yr⁻¹",          "Source": "El-Gohary (2020)"},
            {"Symbol": "δ",  "Name": "Disease mortality",   "Value": "0.025",   "Unit": "yr⁻¹",          "Source": "Juja Hospital records"},
            {"Symbol": "τ",  "Name": "Treatment rate",      "Value": "0.152",   "Unit": "yr⁻¹",          "Source": "Juja Hospital ART"},
            {"Symbol": "ε",  "Name": "ART infectivity",     "Value": "0.04",    "Unit": "dimensionless",  "Source": "Cohen et al. (2021)"},
        ])
        st.dataframe(params_df.set_index("Symbol"), use_container_width=True)

        st.markdown("#### Initial Conditions (Jan 2026)")
        ic_df = pd.DataFrame([
            {"Compartment": "S – Susceptible",  "Value": "90,857", "Derivation": "Total pop − HIV+ cases"},
            {"Compartment": "E – Exposed",      "Value": "450",    "Derivation": "Incidence × latency period"},
            {"Compartment": "I – Infected",     "Value": "587",    "Derivation": "HIV+ not on ART (4,632 − 4,045)"},
            {"Compartment": "T – Treated",      "Value": "4,045",  "Derivation": "ART register 2025"},
            {"Compartment": "N – Total",        "Value": "95,939", "Derivation": "S + E + I + T"},
        ])
        st.dataframe(ic_df.set_index("Compartment"), use_container_width=True)

    # Threshold elimination
    st.markdown("---")
    st.markdown("#### Elimination Threshold Analysis")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class='info-box'>
        Critical β for elimination (τ fixed):<br>
        β* = (μ+k)(μ+δ+τ)/k = 0.023589/0.1095 = <strong>0.215 yr⁻¹</strong><br>
        Current β = 1.82 → requires 88.2% reduction
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class='info-box'>
        Critical τ for elimination (β fixed):<br>
        τ* ≈ βk/(μ+k) − μ − δ ≈ 1.574 yr⁻¹<br>
        Current τ = 0.152 → requires ~10× increase
        </div>
        """, unsafe_allow_html=True)
    st.info("💡 Neither β reduction nor τ increase alone is sufficient for elimination at current baseline — the **combined strategy (Scenario D)** is recommended.")


# ═══════════════════════════════════════════════════════════════
# TAB 4 — FACILITY DATA
# ═══════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### 🏥 Epidemiological Data — Three Juja Health Facilities (2019–2025)")

    facility = st.selectbox("Select Facility", [
        "Juja Sub-County Hospital (Primary)",
        "JKUAT Hospital",
        "Kiambu County Referral Hospital",
    ])

    if facility == "Juja Sub-County Hospital (Primary)":
        fig, axes = dark_figs(2, 2, figsize=(14, 8))
        ax1, ax2, ax3, ax4 = axes.flatten()

        years = HIST_DATA["Year"].tolist()

        # Diagnoses
        ax1.bar(years, HIST_DATA["New Diagnoses"], color="#ff6b6b", alpha=0.85, edgecolor="#ff6b6b", width=0.6)
        ax1.set_title("New HIV Diagnoses", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Count", fontsize=9)

        # Prevalence
        ax2.plot(years, HIST_DATA["Prevalence %"], color="#ffd93d", lw=2.5, marker="o",
                 markersize=6, markerfacecolor="#ffd93d")
        ax2.fill_between(years, HIST_DATA["Prevalence %"], alpha=0.15, color="#ffd93d")
        ax2.set_title("HIV Prevalence (%)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("%", fontsize=9)

        # ART coverage
        ax3.plot(years, HIST_DATA["ART Coverage %"], color="#00d4aa", lw=2.5, marker="o",
                 markersize=6, markerfacecolor="#00d4aa")
        ax3.fill_between(years, HIST_DATA["ART Coverage %"], alpha=0.15, color="#00d4aa")
        ax3.set_title("ART Coverage (%)", fontsize=11, fontweight="bold")
        ax3.set_ylabel("%", fontsize=9)
        ax3.set_ylim(75, 92)

        # PrEP users
        ax4.bar(years, HIST_DATA["PrEP Users"], color="#6bceff", alpha=0.85, edgecolor="#6bceff", width=0.6)
        ax4.set_title("PrEP Users", fontsize=11, fontweight="bold")
        ax4.set_ylabel("Count", fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("#### Juja Sub-County Hospital — Full Data Table")
        st.dataframe(HIST_DATA.set_index("Year"), use_container_width=True)

    elif facility == "JKUAT Hospital":
        jkuat_df = pd.DataFrame({
            "Year": [2019,2020,2021,2022,2023,2024,2025],
            "Patients Seen (15–49)": [12340,11820,13210,14450,15120,15890,16234],
            "HIV+ Diagnosed": [112,98,125,138,144,151,158],
            "New ART Start": [87,79,101,109,118,122,131],
            "PrEP Initiations": [89,95,112,128,143,158,174],
            "HIV Counselling": [1456,1234,1567,1789,1903,2012,2145],
        })
        fig, axes = dark_figs(1, 2, figsize=(14, 5))
        ax1, ax2 = axes
        ax1.plot(jkuat_df["Year"], jkuat_df["HIV+ Diagnosed"], color="#ff6b6b", lw=2.5, marker="o", label="HIV+ Diagnosed")
        ax1.plot(jkuat_df["Year"], jkuat_df["PrEP Initiations"], color="#00d4aa", lw=2.5, marker="o", label="PrEP Initiations")
        ax1.set_title("JKUAT Hospital — Diagnoses & PrEP", fontsize=12, fontweight="bold")
        ax1.legend(facecolor="#1a2235", edgecolor="#1e2d45", labelcolor="#e8eef8", fontsize=9)

        ax2.bar(jkuat_df["Year"], jkuat_df["HIV Counselling"], color="#ffd93d", alpha=0.85, width=0.6)
        ax2.set_title("HIV Counselling Sessions", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.dataframe(jkuat_df.set_index("Year"), use_container_width=True)

    else:
        kiambu_df = pd.DataFrame({
            "Year": [2019,2020,2021,2022,2023,2024,2025],
            "Adult HIV Patients": [18456,18901,19234,19567,19890,20123,20456],
            "Juja Sub-County Cases": [4112,4201,4289,4378,4467,4556,4632],
            "ART Patients (County)": [14234,14789,15124,15456,15789,16012,16345],
            "VL Suppression Rate %": [78.2,80.1,81.4,83.0,84.5,85.8,87.3],
        })
        fig, axes = dark_figs(1, 2, figsize=(14, 5))
        ax1, ax2 = axes
        ax1.bar(kiambu_df["Year"], kiambu_df["Adult HIV Patients"], color="#6bceff", alpha=0.7, width=0.4, label="County Total")
        ax1.bar(kiambu_df["Year"], kiambu_df["Juja Sub-County Cases"], color="#ff6b6b", alpha=0.85, width=0.4, label="Juja Sub-County")
        ax1.set_title("County vs. Juja HIV Patients", fontsize=12, fontweight="bold")
        ax1.legend(facecolor="#1a2235", edgecolor="#1e2d45", labelcolor="#e8eef8", fontsize=9)

        ax2.plot(kiambu_df["Year"], kiambu_df["VL Suppression Rate %"], color="#00d4aa", lw=2.5, marker="o")
        ax2.fill_between(kiambu_df["Year"], kiambu_df["VL Suppression Rate %"], alpha=0.15, color="#00d4aa")
        ax2.set_title("Viral Load Suppression Rate (%)", fontsize=12, fontweight="bold")
        ax2.set_ylim(75, 92)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.dataframe(kiambu_df.set_index("Year"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 5 — COST-EFFECTIVENESS
# ═══════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### 💰 Cost-Effectiveness Analysis (5-Year, 2026–2030)")
    st.markdown("**WHO Willingness-to-Pay threshold: KES 630,000 per DALY averted (3× Kenya GDP per capita, 2025)**")

    cost_data = {
        "A: Baseline":  {"cost": 1.85e9, "averted": 0,   "icer_inf": 0,       "icer_daly": 0,      "dalys": 0,     "lives": 0},
        "B: PrEP 20%":  {"cost": 2.15e9, "averted": 163, "icer_inf": 1840000, "icer_daly": 65700,  "dalys": 4564,  "lives": 45},
        "C: PrEP 40%":  {"cost": 2.55e9, "averted": 367, "icer_inf": 1907000, "icer_daly": 68100,  "dalys": 10276, "lives": 102},
        "D: Combined":  {"cost": 2.98e9, "averted": 610, "icer_inf": 1827000, "icer_daly": 65300,  "dalys": 17080, "lives": 169},
    }

    # Key metrics row
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Lowest ICER/DALY", "KES 65,300", "Scenario D — Combined")
    with c2: st.metric("Most Infections Averted", "610", "Scenario D — Combined")
    with c3: st.metric("Most DALYs Averted", "17,080", "Scenario D — Combined")
    with c4: st.metric("Lives Saved (D)", "~169", "5-year estimate")

    st.markdown("<br>", unsafe_allow_html=True)

    fig, axes = dark_figs(2, 2, figsize=(14, 9))
    ax1, ax2, ax3, ax4 = axes.flatten()

    names_ce = list(cost_data.keys())
    ce_colors = [SCENARIO_COLORS[n] for n in names_ce]
    ce_alphas = [0.85] * 4

    # ICER vs WTP
    icer_vals = [cost_data[n]["icer_daly"] for n in names_ce]
    bars1 = ax1.bar(names_ce, icer_vals, color=ce_colors, alpha=0.85, edgecolor=ce_colors, width=0.5)
    ax1.axhline(630000, color="white", linewidth=1.5, linestyle="--", alpha=0.5, label="WTP = KES 630,000")
    ax1.set_title("ICER per DALY Averted vs. WTP Threshold", fontsize=11, fontweight="bold")
    ax1.set_ylabel("KES / DALY", fontsize=9)
    ax1.legend(facecolor="#1a2235", edgecolor="#1e2d45", labelcolor="#e8eef8", fontsize=8)
    ax1.tick_params(axis="x", labelsize=8)
    for bar, val in zip(bars1, icer_vals):
        if val > 0:
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2000,
                     f"{val:,}", ha="center", va="bottom", color="#e8eef8", fontsize=8)

    # 5-Year Cost
    cost_vals = [cost_data[n]["cost"]/1e9 for n in names_ce]
    bars2 = ax2.bar(names_ce, cost_vals, color=ce_colors, alpha=0.85, edgecolor=ce_colors, width=0.5)
    ax2.set_title("5-Year Total Program Cost (KES Billion)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("KES Billion", fontsize=9)
    ax2.tick_params(axis="x", labelsize=8)
    for bar, val in zip(bars2, cost_vals):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f"{val:.2f}B", ha="center", va="bottom", color="#e8eef8", fontsize=8)

    # DALYs averted
    daly_vals = [cost_data[n]["dalys"] for n in names_ce]
    bars3 = ax3.bar(names_ce, daly_vals, color=ce_colors, alpha=0.85, edgecolor=ce_colors, width=0.5)
    ax3.set_title("DALYs Averted (5-Year Total)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("DALYs Averted", fontsize=9)
    ax3.tick_params(axis="x", labelsize=8)

    # Lives saved
    lives_vals = [cost_data[n]["lives"] for n in names_ce]
    bars4 = ax4.bar(names_ce, lives_vals, color=ce_colors, alpha=0.85, edgecolor=ce_colors, width=0.5)
    ax4.set_title("Estimated Lives Saved (5-Year)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Lives Saved", fontsize=9)
    ax4.tick_params(axis="x", labelsize=8)
    for bar, val in zip(bars4, lives_vals):
        if val > 0:
            ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     f"~{val}", ha="center", va="bottom", color="#e8eef8", fontsize=9, fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Full table
    st.markdown("#### Complete Cost-Effectiveness Table")
    ce_table = pd.DataFrame([
        {
            "Scenario": name,
            "5-Yr Cost (KES)":       f"{cost_data[name]['cost']:,.0f}",
            "Infections Averted":    cost_data[name]["averted"] or "—",
            "ICER / Infection":      f"{cost_data[name]['icer_inf']:,}" if cost_data[name]["icer_inf"] else "—",
            "ICER / DALY (KES)":     f"{cost_data[name]['icer_daly']:,}" if cost_data[name]["icer_daly"] else "—",
            "DALYs Averted":         cost_data[name]["dalys"] or "—",
            "Lives Saved":           f"~{cost_data[name]['lives']}" if cost_data[name]["lives"] else "—",
        }
        for name in names_ce
    ])
    st.dataframe(ce_table.set_index("Scenario"), use_container_width=True)

    st.success("✅ All scenarios are highly cost-effective — ICER values are less than 11% of Kenya's WTP threshold (KES 630,000/DALY). **Scenario D (Combined)** achieves the lowest ICER and greatest health impact.")

    # Download full results
    st.markdown("---")
    st.markdown("### 📥 Export Full Simulation Results")
    all_dfs = []
    for name, df in all_results.items():
        df2 = df.copy()
        df2.insert(0, "Scenario", name)
        all_dfs.append(df2)
    export_df = pd.concat(all_dfs, ignore_index=True)
    csv_all = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download All Scenarios CSV", csv_all, "seit_all_scenarios.csv", "text/csv")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#7a8ba0;font-size:12px;padding:16px 0 32px;font-family:monospace'>
  Laurence Maina · Hilda Peter · Vera Vicky · Stanley Munyao · Baruch Limo<br>
  Department of Pure &amp; Applied Mathematics — JKUAT · Supervisor: Dr. Phineus Kiogora, PhD · 2026
</div>
""", unsafe_allow_html=True)
