import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64

st.set_page_config(layout="wide", page_title="PD / Survival Visualizer")

# -------------------------
# Helper functions
# -------------------------
def series_from_manual(manual_vals, n):
    """Take up to 5 manual values. If n > len(manual_vals), extend by repeating last value."""
    vals = list(manual_vals)
    if len(vals) == 0:
        vals = [0.01] * n
    if len(vals) >= n:
        return np.array(vals[:n])
    # extend by repeating last value
    last = vals[-1]
    vals_extended = vals + [last] * (n - len(vals))
    return np.array(vals_extended)


def series_from_ttc_with_noise(ttc, n, noise_level, random_seed=None):
    """Generate PIT series around TTC with normally-distributed noise, truncated to [0,1]."""
    rng = np.random.RandomState(random_seed)
    noise = rng.normal(loc=0.0, scale=noise_level, size=n)
    series = np.clip(ttc + noise, 0.0, 0.9999)  # avoid exactly 1
    return series


def survival_from_pit_series(pit):
    """S(0)=1. S(i) = product_{k=1..i} (1 - pit[k-1])"""
    n = len(pit)
    S = np.empty(n + 1)  # S[0]..S[n]
    S[0] = 1.0
    for i in range(1, n + 1):
        S[i] = S[i - 1] * (1.0 - pit[i - 1])
    return S  # length n+1


def compute_marginal_and_cumulative(S):
    """Given survival array S[0..n], compute cumulative C[0..n] and marginal m[1..n] for periods 1..n"""
    n = len(S) - 1
    C = 1.0 - S  # vectorized, length n+1
    marginal = np.empty(n + 1)
    marginal[0] = 0.0
    for i in range(1, n + 1):
        marginal[i] = C[i] - C[i - 1]  # = S[i-1]*pit[i]
    return C, marginal  # marginal[1..n] meaningful


def survival_ttc_discrete(ttc, n):
    """Case 2: discrete TTC PD constant p. S(i) = (1 - p)^i for i=0..n"""
    i = np.arange(0, n + 1)
    S = (1.0 - ttc) ** i
    return S


def survival_exponential(lam, n):
    """Case 3: continuous exponential S(t)=exp(-lambda * t) at integer times t=0..n"""
    t = np.arange(0, n + 1)
    S = np.exp(-lam * t)
    return S


def plot_column(title, periods, S, C, marginal, axes_width=4.5, axes_height=3.0):
    """Return matplotlib figure with Survival + Cumulative lines and Marginal bars stacked."""
    fig, ax = plt.subplots(3, 1, figsize=(axes_width, axes_height * 3), constrained_layout=True)

    # Survival
    ax[0].plot(periods, S, marker='o', linewidth=2)
    ax[0].set_title(f"{title} — Survival rate S(i)")
    ax[0].set_xlabel("Period i")
    ax[0].set_ylabel("Survival S(i)")
    ax[0].set_ylim(0.0, 1.0)
    ax[0].grid(True, linestyle="--", alpha=0.4)

    # Cumulative PD
    ax[1].plot(periods, C, marker='o', linewidth=2, color='tab:orange')
    ax[1].set_title(f"{title} — Cumulative PD C(i) = 1 - S(i)")
    ax[1].set_xlabel("Period i")
    ax[1].set_ylabel("Cumulative PD")
    ax[1].set_ylim(0.0, 1.0)
    ax[1].grid(True, linestyle="--", alpha=0.4)

    # Marginal PD (bars)
    ax[2].bar(periods, marginal, alpha=0.7)
    ax[2].set_title(f"{title} — Marginal PD m(i) = C(i)-C(i-1)")
    ax[2].set_xlabel("Period i")
    ax[2].set_ylabel("Marginal PD")
    ax[2].set_ylim(0.0, max(0.01, marginal.max() * 1.1))
    ax[2].grid(True, axis='y', linestyle="--", alpha=0.4)

    return fig


def df_display_table(periods, S, C, marginal):
    df = pd.DataFrame({
        "Period": periods,
        "Survival S(i)": np.round(S, 6),
        "Cumulative PD C(i)": np.round(C, 6),
        "Marginal PD m(i)": np.round(marginal, 6)
    })
    return df


def fig_to_png_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def download_link_for_bytes(data_bytes, filename, label):
    b64 = base64.b64encode(data_bytes).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">{label}</a>'
    return href


# -------------------------
# UI / App
# -------------------------
st.title("📈 PD / Survival Visualizer — Marginal PD, Survival, and Cumulative PD")
st.write("""
This app demonstrates how **Survival rate**, **Cumulative PD**, and **Marginal PD** relate under three modelling approaches:
- **Case 1 (PIT time series)**: arbitrary period-by-period PIT PDs (user input or generated from TTC + noise)
- **Case 2 (Discrete TTC)**: constant TTC PD per period, survival is \((1-p)^i\)
- **Case 3 (Exponential)**: continuous hazard \(S(t)=e^{-\lambda t}\) (discrete observations at integer t)
Each column is independent so you can compare parameter choices side-by-side.
""")

st.markdown("---")

# global controls
n_periods = st.sidebar.slider("Number of periods (n)", 3, 30, 10)
show_tables = st.sidebar.checkbox("Show numeric tables", value=True)
random_seed = st.sidebar.number_input("Random seed (for noise reproducibility)", value=42, step=1)

# Create three columns (Case1, Case2, Case3)
col_case1, col_case2, col_case3 = st.columns(3)

# -------------------------
# CASE 1: PIT time series
# -------------------------
with col_case1:
    st.header("Case 1 — PIT Time Series")
    st.write("Survival defined by product:  S(0)=1;  S(i)=S(i-1)*(1 - PiT_PD(i))")
    st.markdown("**Derivation (discrete PIT):**")
    st.latex(r"S(0)=1")
    st.latex(r"S(i)=\prod_{k=1}^i (1 - \mathrm{PiT\_PD}(k))")
    st.latex(r"C(i)=1-S(i)")
    st.latex(r"\text{Marginal } m(i)=C(i)-C(i-1)=\mathrm{PiT\_PD}(i)\times S(i-1)")

    mode = st.selectbox("Input mode for PIT series", ["Default example series",
                                                     "Manual (enter up to 5 PIT PDs)",
                                                     "TTC PD + noise generator"], key="case1_mode")

    if mode == "Default example series":
        # an illustrative example series
        pit_series_case1 = np.array([0.02, 0.03, 0.05, 0.04, 0.03])
        st.info(f"Using default example PIT series (first 5 shown): {pit_series_case1.tolist()}")
        pit_vals = series_from_manual(list(pit_series_case1), n_periods)

    elif mode == "Manual (enter up to 5 PIT PDs)":
        st.write("Enter up to 5 PIT PDs (values between 0 and 1). If n > 5, last value is repeated to fill periods.")
        m1 = st.number_input("PIT PD 1", min_value=0.0, max_value=1.0, value=0.02, key="c1_m1")
        m2 = st.number_input("PIT PD 2", min_value=0.0, max_value=1.0, value=0.03, key="c1_m2")
        m3 = st.number_input("PIT PD 3", min_value=0.0, max_value=1.0, value=0.05, key="c1_m3")
        m4 = st.number_input("PIT PD 4", min_value=0.0, max_value=1.0, value=0.04, key="c1_m4")
        m5 = st.number_input("PIT PD 5", min_value=0.0, max_value=1.0, value=0.03, key="c1_m5")
        manual_list = [m1, m2, m3, m4, m5]
        pit_vals = series_from_manual(manual_list, n_periods)
        st.write(f"PIT series used (length {n_periods}):")
        st.write(np.round(pit_vals, 6).tolist())

    else:  # TTC PD + noise
        ttc = st.number_input("TTC PD (base)", min_value=0.0, max_value=1.0, value=0.03, key="c1_ttc")
        noise = st.slider("Noise (std dev)", 0.0, 0.2, 0.02, key="c1_noise")
        pit_vals = series_from_ttc_with_noise(ttc, n_periods, noise, random_seed)
        st.write(f"Generated PIT series (TTC={ttc}, noise={noise}):")
        st.write(np.round(pit_vals, 6).tolist())

    # compute survival, cumulative, marginal
    S_case1 = survival_from_pit_series(pit_vals)  # length n+1
    C_case1, m_case1 = compute_marginal_and_cumulative(S_case1)

    periods = np.arange(0, n_periods + 1)
    periods_for_marginal = np.arange(0, n_periods + 1)

    # prepare plots and table
    fig1 = plot_column("Case 1 (PIT series)", periods, S_case1, C_case1, m_case1)
    st.pyplot(fig1)

    if show_tables:
        df1 = df_display_table(periods, S_case1, C_case1, m_case1)
        st.dataframe(df1)

    # small explanation
    st.markdown("**Notes (Case 1):** PIT series are per-period default probabilities. The survival at period i is the product of survival factors up to i. Marginal PD in period i equals PIT(i) * S(i-1).")

# -------------------------
# CASE 2: Discrete TTC PD
# -------------------------
with col_case2:
    st.header("Case 2 — Discrete TTC PD (constant per period)")
    st.markdown("Assume constant TTC PD per period:  PiT_PD(i) = p (constant).")
    st.markdown("**Derivation (discrete constant hazard):**")
    st.latex(r"S(i)=\prod_{k=1}^i (1 - p) = (1-p)^i")
    st.latex(r"C(i)=1-(1-p)^i")
    st.latex(r"\text{Marginal } m(i) = (1-p)^{\,i-1}\,p")

    ttc2 = st.number_input("TTC PD (p)", min_value=0.0, max_value=1.0, value=0.03, key="case2_ttc")
    # Optionally allow different parameterization: allow "per-period varying p"? But spec says TTC PD.
    S_case2 = survival_ttc_discrete(ttc2, n_periods)
    C_case2, m_case2 = compute_marginal_and_cumulative(S_case2)

    periods = np.arange(0, n_periods + 1)
    fig2 = plot_column("Case 2 (Discrete TTC)", periods, S_case2, C_case2, m_case2)
    st.pyplot(fig2)

    if show_tables:
        df2 = df_display_table(periods, S_case2, C_case2, m_case2)
        st.dataframe(df2)

    st.markdown("**Notes (Case 2):** With constant TTC PD p, marginal PD simplifies to \\((1-p)^{i-1} p\\). This is the binomial / geometric style discrete hazard.")


# -------------------------
# CASE 3: Exponential (continuous hazard)
# -------------------------
with col_case3:
    st.header("Case 3 — Exponential (continuous hazard)")
    st.markdown("Assume continuous-time hazard with rate \\(\\lambda\\). Observing at integer times t=0,1,2,...")
    st.markdown("**Derivation (exponential):**")
    st.latex(r"S(t)=e^{-\lambda t}")
    st.latex(r"C(t)=1-e^{-\lambda t}")
    st.latex(r"\text{Discrete marginal between }(i-1,i]:\; m(i)=S(i-1)-S(i)=e^{-\lambda (i-1)}(1-e^{-\lambda})")

    lam = st.number_input("Lambda (λ) — hazard (continuous)", min_value=0.0001, max_value=10.0, value=0.03, key="case3_lambda")
    S_case3 = survival_exponential(lam, n_periods)
    C_case3, m_case3 = compute_marginal_and_cumulative(S_case3)

    periods = np.arange(0, n_periods + 1)
    fig3 = plot_column("Case 3 (Exponential)", periods, S_case3, C_case3, m_case3)
    st.pyplot(fig3)

    if show_tables:
        df3 = df_display_table(periods, S_case3, C_case3, m_case3)
        st.dataframe(df3)

    st.markdown("**Notes (Case 3):** Lambda controls the continuous hazard. For small λ, exponential approximates small discrete hazards. For comparison with TTC PD p, λ ≈ -ln(1-p) (so that 1 - e^{-λ} ≈ p).")

st.markdown("---")
st.subheader("Quick comparisons & tips")
st.write("""
- **Comparing Case 2 and Case 3:** If you want the discrete per-period marginal to match approximately, set \\(\\lambda\\) so that \\(1-e^{-\\lambda} = p\\) (i.e. \\(\\lambda = -\\ln(1-p)\\)).
- **Interpreting Marginal PD:** It is the *probability of default in that specific period*, conditional on survival to the period start.
- **Cumulative PD** increases monotonically and approaches 1 as periods become large (depending on the hazard).
""")

st.markdown("### Export plots")
st.write("You can download PNG snapshots of each column's figure below.")

colA, colB, colC = st.columns(3)
with colA:
    bts = fig_to_png_bytes(fig1)
    st.markdown(download_link_for_bytes(bts, "case1_pds.png", "Download Case 1 plots as PNG"), unsafe_allow_html=True)

with colB:
    bts2 = fig_to_png_bytes(fig2)
    st.markdown(download_link_for_bytes(bts2, "case2_pds.png", "Download Case 2 plots as PNG"), unsafe_allow_html=True)

with colC:
    bts3 = fig_to_png_bytes(fig3)
    st.markdown(download_link_for_bytes(bts3, "case3_pds.png", "Download Case 3 plots as PNG"), unsafe_allow_html=True)
