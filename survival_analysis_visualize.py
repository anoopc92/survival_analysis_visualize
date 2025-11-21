import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64

st.set_page_config(layout="wide", page_title="PD / Survival Comparator")

# -------------------------
# Helper functions
# -------------------------
def series_from_manual(manual_vals, n):
    vals = list(manual_vals)
    if len(vals) == 0:
        vals = [0.01] * n
    if len(vals) >= n:
        return np.array(vals[:n])
    last = vals[-1]
    return np.array(vals + [last]*(n-len(vals)))

def series_from_ttc_with_noise(ttc, n, noise_level, random_seed=None):
    rng = np.random.RandomState(random_seed)
    noise = rng.normal(0, noise_level, n)
    return np.clip(ttc + noise, 0.0, 0.9999)

def series_from_csv(uploaded_file, n):
    df = pd.read_csv(uploaded_file)
    # Take first numeric column
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            series = df[col].values
            return series_from_manual(series.tolist(), n)
    st.warning("No numeric column found in CSV; using default series.")
    return series_from_manual([], n)

def survival_from_pit_series(pit):
    n = len(pit)
    S = np.empty(n+1)
    S[0]=1.0
    for i in range(1,n+1):
        S[i]=S[i-1]*(1-pit[i-1])
    return S

def compute_marginal_and_cumulative(S):
    n = len(S)-1
    C = 1.0 - S
    m = np.empty(n+1)
    m[0]=0
    for i in range(1,n+1):
        m[i] = C[i]-C[i-1]
    return C,m

def survival_ttc_discrete(ttc,n):
    return (1-ttc)**np.arange(0,n+1)

def survival_exponential(lam,n):
    return np.exp(-lam*np.arange(0,n+1))

def plot_comparison_row(title, periods, data_list, labels, ylabel):
    cols = st.columns(len(data_list))
    figs=[]
    for col,d,label in zip(cols,data_list,labels):
        fig, ax = plt.subplots(figsize=(4.5,3))
        ax.plot(periods,d,marker='o',linewidth=2)
        ax.set_title(f"{title} — {label}")
        ax.set_xlabel("Period i")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0,1.0)
        ax.grid(True, linestyle="--", alpha=0.4)
        col.pyplot(fig)
        figs.append(fig)
    return figs

def plot_marginal_row(title, periods, marg_list, labels):
    cols = st.columns(len(marg_list))
    figs=[]
    for col,m,label in zip(cols,marg_list,labels):
        fig, ax = plt.subplots(figsize=(4.5,3))
        ax.bar(periods,m,alpha=0.7)
        ax.set_title(f"{title} — {label}")
        ax.set_xlabel("Period i")
        ax.set_ylabel("Marginal PD")
        ax.set_ylim(0.0,max(0.01, max([m_.max() for m_ in marg_list])*1.1))
        ax.grid(True,axis='y', linestyle="--", alpha=0.4)
        col.pyplot(fig)
        figs.append(fig)
    return figs

def df_display_table(periods,S_list,C_list,m_list,labels):
    dfs=[]
    for S,C,m,label in zip(S_list,C_list,m_list,labels):
        df = pd.DataFrame({
            "Period": periods,
            f"Survival S(i) [{label}]": np.round(S,6),
            f"Cumulative PD C(i) [{label}]": np.round(C,6),
            f"Marginal PD m(i) [{label}]": np.round(m,6)
        })
        dfs.append(df)
    return dfs

# -------------------------
# Sidebar / Global controls
# -------------------------
st.sidebar.header("Global settings")
n_periods = st.sidebar.slider("Number of periods (n)",3,30,10)
show_tables = st.sidebar.checkbox("Show numeric tables", value=True)
random_seed = st.sidebar.number_input("Random seed (for noise)",value=42,step=1)

compare_mode = st.sidebar.radio("Comparison mode",["Three approaches","Same approach different parameters"])

# Synchronization options
st.sidebar.markdown("### Synchronization options")
sync_ttc = st.sidebar.checkbox("Sync TTC PD across columns", value=False)
sync_lambda = st.sidebar.checkbox("Sync Lambda to TTC PD (λ=-ln(1-p))", value=False)

# -------------------------
# Determine columns & labels
# -------------------------
if compare_mode=="Three approaches":
    approach_labels=["Case 1 PIT","Case 2 TTC","Case 3 Exponential"]
else:
    approach_options = ["PIT series","TTC","Exponential"]
    chosen_approach = st.sidebar.selectbox("Choose approach for all columns",approach_options)
    approach_labels = [f"{chosen_approach} #{i+1}" for i in range(3)]

# -------------------------
# Prepare input for each column
# -------------------------
pit_series_list=[]
ttc_list=[]
lambda_list=[]

for i in range(3):
    st.sidebar.markdown(f"### Column {i+1}: {approach_labels[i]}")
    if compare_mode=="Three approaches":
        # Case1
        if i==0:
            input_mode = st.sidebar.selectbox(f"Column {i+1} PIT input mode", ["Default","Manual","TTC+noise","CSV upload"], key=f"c1_mode")
            if input_mode=="Default":
                pit_series = series_from_manual([0.02,0.03,0.05,0.04,0.03],n_periods)
            elif input_mode=="Manual":
                vals = []
                for j in range(5):
                    val = st.sidebar.number_input(f"Period {j+1}",0.0,1.0,0.02,key=f"c1_manual_{j}")
                    vals.append(val)
                pit_series = series_from_manual(vals,n_periods)
            elif input_mode=="TTC+noise":
                ttc_val = st.sidebar.number_input(f"TTC PD for Column {i+1}",0.0,1.0,0.03,key=f"c1_ttc")
                noise = st.sidebar.slider(f"Noise std dev for Column {i+1}",0.0,0.2,0.02,key=f"c1_noise")
                pit_series = series_from_ttc_with_noise(ttc_val,n_periods,noise,random_seed)
            else: # CSV
                uploaded_file = st.sidebar.file_uploader(f"Upload CSV PIT for Column {i+1}", type=['csv'], key=f"c1_csv")
                if uploaded_file is not None:
                    pit_series = series_from_csv(uploaded_file,n_periods)
                else:
                    pit_series = series_from_manual([0.02,0.03,0.05,0.04,0.03],n_periods)
            pit_series_list.append(pit_series)
        elif i==1:
            ttc_val = st.sidebar.number_input(f"TTC PD for Column {i+1}",0.0,1.0,0.03,key=f"c2_ttc")
            ttc_list.append(ttc_val)
        else:
            lam_val = st.sidebar.number_input(f"Lambda for Column {i+1}",0.0001,10.0,0.03,key=f"c3_lam")
            lambda_list.append(lam_val)
    else:
        # Same approach different params
        if chosen_approach=="PIT series":
            input_mode = st.sidebar.selectbox(f"Column {i+1} PIT input mode", ["Default","Manual","TTC+noise","CSV upload"], key=f"c1_mode_{i}")
            if input_mode=="Default":
                pit_series = series_from_manual([0.02,0.03,0.05,0.04,0.03],n_periods)
            elif input_mode=="Manual":
                vals = []
                for j in range(5):
                    val = st.sidebar.number_input(f"Period {j+1} Col{i+1}",0.0,1.0,0.02,key=f"manual_{i}_{j}")
                    vals.append(val)
                pit_series = series_from_manual(vals,n_periods)
            elif input_mode=="TTC+noise":
                ttc_val = st.sidebar.number_input(f"TTC PD for Column {i+1}",0.0,1.0,0.03,key=f"ttc_{i}")
                noise = st.sidebar.slider(f"Noise std dev for Column {i+1}",0.0,0.2,0.02,key=f"noise_{i}")
                pit_series = series_from_ttc_with_noise(ttc_val,n_periods,noise,random_seed)
            else:
                uploaded_file = st.sidebar.file_uploader(f"Upload CSV PIT for Column {i+1}", type=['csv'], key=f"csv_{i}")
                if uploaded_file is not None:
                    pit_series = series_from_csv(uploaded_file,n_periods)
                else:
                    pit_series = series_from_manual([0.02,0.03,0.05,0.04,0.03],n_periods)
            pit_series_list.append(pit_series)
        elif chosen_approach=="TTC":
            ttc_val = st.sidebar.number_input(f"TTC PD for Column {i+1}",0.0,1.0,0.03,key=f"ttc_{i}")
            ttc_list.append(ttc_val)
        else:
            lam_val = st.sidebar.number_input(f"Lambda for Column {i+1}",0.0001,10.0,0.03,key=f"lam_{i}")
            lambda_list.append(lam_val)

# -------------------------
# Sync options
# -------------------------
if sync_ttc and len(ttc_list)>0:
    ttc_global = st.sidebar.number_input("Global TTC PD",0.0,1.0,0.03,key="global_ttc")
    ttc_list = [ttc_global]*len(ttc_list)
    # update PIT columns if using TTC+noise mode
    for idx,pit_series in enumerate(pit_series_list):
        pit_series_list[idx] = series_from_ttc_with_noise(ttc_global,n_periods,0.02,random_seed)

if sync_lambda and len(ttc_list)>0:
    lambda_list = [-np.log(1-p) for p in ttc_list]

# -------------------------
# Compute S, C, m for each column
# -------------------------
S_list=[]
C_list=[]
m_list=[]
periods=np.arange(0,n_periods+1)

for i in range(3):
    if compare_mode=="Three approaches":
        if i==0:
            pit = pit_series_list[0]
            S=survival_from_pit_series(pit)
            C,m=compute_marginal_and_cumulative(S)
        elif i==1:
            ttc = ttc_list[0]
            S=survival_ttc_discrete(ttc,n_periods)
            C,m=compute_marginal_and_cumulative(S)
        else:
            lam = lambda_list[0]
            S=survival_exponential(lam,n_periods)
            C,m=compute_marginal_and_cumulative(S)
    else:
        if chosen_approach=="PIT series":
            pit = pit_series_list[i]
            S=survival_from_pit_series(pit)
            C,m=compute_marginal_and_cumulative(S)
        elif chosen_approach=="TTC":
            ttc=ttc_list[i]
            S=survival_ttc_discrete(ttc,n_periods)
            C,m=compute_marginal_and_cumulative(S)
        else:
            lam = lambda_list[i]
            S=survival_exponential(lam,n_periods)
            C,m=compute_marginal_and_cumulative(S)
    S_list.append(S)
    C_list.append(C)
    m_list.append(m)

# -------------------------
# Display derivation formulas with explanations
# -------------------------
st.markdown("## Derivation Formulas and Term Explanations")

with st.expander("Show formulas and explanations"):
    if compare_mode=="Three approaches":
        st.markdown(r"""
**Case 1 (PIT series)**  
- \(S(0) = 1\) : initial survival probability  
- \(S(i) = \prod_{k=1}^{i} (1 - \mathrm{PiT\_PD}(k))\) : survival at period \(i\) as product of survival factors  
- \(C(i) = 1 - S(i)\) : cumulative probability of default up to period \(i\)  
- \(m(i) = C(i) - C(i-1) = \mathrm{PiT\_PD}(i) \cdot S(i-1)\) : marginal PD in period \(i\), conditional on survival  

**Case 2 (Discrete TTC)**  
- \(S(i) = (1-p)^i\) : constant TTC PD per period  
- \(C(i) = 1 - (1-p)^i\) : cumulative PD  
- \(m(i) = (1-p)^{i-1} \cdot p\) : marginal PD  

**Case 3 (Exponential hazard)**  
- \(S(t) = e^{-\lambda t}\) : survival in continuous time  
- \(C(t) = 1 - e^{-\lambda t}\) : cumulative PD  
- \(m(i) = S(i-1) - S(i) = e^{-\lambda (i-1)} (1 - e^{-\lambda})\) : discrete marginal PD between i-1 and i
""",unsafe_allow_html=True)
    else:
        if chosen_approach=="PIT series":
            st.markdown(r"""
**PIT series**  
- \(S(0) = 1\) : initial survival probability  
- \(S(i) = \prod_{k=1}^{i} (1 - \mathrm{PiT\_PD}(k))\) : survival at period \(i\)  
- \(C(i) = 1 - S(i)\) : cumulative PD  
- \(m(i) = C(i) - C(i-1) = \mathrm{PiT\_PD}(i) \cdot S(i-1)\) : marginal PD  
""",unsafe_allow_html=True)
        elif chosen_approach=="TTC":
            st.markdown(r"""
**Discrete TTC**  
- \(S(i) = (1-p)^i\) : constant TTC PD per period  
- \(C(i) = 1 - (1-p)^i\) : cumulative PD  
- \(m(i) = (1-p)^{i-1} \cdot p\) : marginal PD  
""",unsafe_allow_html=True)
        else:
            st.markdown(r"""
**Exponential hazard**  
- \(S(t) = e^{-\lambda t}\) : survival in continuous time  
- \(C(t) = 1 - e^{-\lambda t}\) : cumulative PD  
- \(m(i) = S(i-1) - S(i) = e^{-\lambda (i-1)} (1 - e^{-\lambda})\) : discrete marginal PD  
""",unsafe_allow_html=True)

# -------------------------
# Plot rows
# -------------------------
st.markdown("## Survival Rate Comparison")
plot_comparison_row("Survival",periods,S_list,approach_labels,"Survival S(i)")

st.markdown("## Cumulative PD Comparison")
plot_comparison_row("Cumulative PD",periods,C_list,approach_labels,"Cumulative PD")

st.markdown("## Marginal PD Comparison")
plot_marginal_row("Marginal PD",periods,m_list,approach_labels)

# -------------------------
# Show numeric tables
# -------------------------
if show_tables:
    st.markdown("## Tables per column")
    dfs = df_display_table(periods,S_list,C_list,m_list,approach_labels)
    for df,label in zip(dfs,approach_labels):
        st.markdown(f"**{label}**")
        st.dataframe(df)
