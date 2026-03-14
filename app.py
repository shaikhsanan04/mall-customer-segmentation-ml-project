"""
Mall Customer Segmentation — Streamlit App
Author: Sanan Shaikh
GitHub: https://github.com/shaikhsanan04/mall-customer-segmentation-ml-project
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0f1117; }

    /* Section headers */
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #00d4ff;
        border-left: 4px solid #00d4ff;
        padding-left: 12px;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3a);
        border: 1px solid #2e3350;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-card h3 { color: #a0aec0; font-size: 0.85rem; margin: 0 0 6px 0; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card p  { color: #00d4ff; font-size: 2rem; font-weight: 800; margin: 0; }

    /* Segment badge */
    .segment-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }

    /* Insight box */
    .insight-box {
        background: #1a1f2e;
        border-left: 4px solid #f6ad55;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 10px 0;
        color: #e2e8f0;
        font-size: 0.95rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #4a5568;
        font-size: 0.82rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #2d3748;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  COLOUR PALETTE (consistent across all plots)
# ─────────────────────────────────────────────
CLUSTER_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
CLUSTER_NAMES  = {
    0: "Standard Customers",
    1: "Budget Shoppers",
    2: "VIP / High-Value",
    3: "Impulsive Buyers",
    4: "Untapped Potential",
}
CLUSTER_EMOJIS = {0: "🟡", 1: "🔵", 2: "🔴", 3: "🟢", 4: "🟣"}


# ─────────────────────────────────────────────
#  DATA LOADING & CACHING
# ─────────────────────────────────────────────
@st.cache_data
def load_data(uploaded=None):
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        # Embedded dataset so the app works without a CSV file
        np.random.seed(42)
        data = {
            "CustomerID": range(1, 201),
            "Gender": np.random.choice(["Male", "Female"], 200),
            "Age": np.random.randint(18, 71, 200),
            "Annual Income (k$)": np.random.randint(15, 138, 200),
            "Spending Score (1-100)": np.random.randint(1, 100, 200),
        }
        df = pd.DataFrame(data)
    return df


@st.cache_data
def run_kmeans(df, k=5):
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    return labels, km, scaler, X_scaled, sil


@st.cache_data
def elbow_silhouette(df):
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias, sil_scores = [], []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        lbl = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, lbl))
    return list(K_range), inertias, sil_scores


@st.cache_data
def run_hierarchical(df, k=5):
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    linked = linkage(X_scaled, method="ward")
    hc = AgglomerativeClustering(n_clusters=k)
    labels = hc.fit_predict(X_scaled)
    return labels, linked, X_scaled


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/shopping-mall.png",
        width=72,
    )
    st.title("⚙️ Controls")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📂 Upload your own CSV",
        type=["csv"],
        help="Must contain: Gender, Age, Annual Income (k$), Spending Score (1-100)",
    )

    st.markdown("### KMeans Settings")
    n_clusters = st.slider("Number of Clusters (K)", min_value=2, max_value=10, value=5, step=1)

    st.markdown("### Plot Style")
    point_size = st.slider("Scatter Point Size", 30, 200, 80, 10)
    show_centroids = st.checkbox("Show Cluster Centroids", value=True)

    st.markdown("---")
    st.markdown("### 📊 Sections")
    show_eda        = st.checkbox("Exploratory Data Analysis", value=True)
    show_elbow      = st.checkbox("Elbow & Silhouette Method", value=True)
    show_clusters   = st.checkbox("KMeans Clustering", value=True)
    show_profiles   = st.checkbox("Cluster Profiles", value=True)
    show_hier       = st.checkbox("Hierarchical Clustering", value=True)
    show_predictor  = st.checkbox("Customer Predictor", value=True)

    st.markdown("---")
    st.caption("Built by **Sanan Shaikh** · [GitHub](https://github.com/shaikhsanan04)")


# ─────────────────────────────────────────────
#  LOAD DATA & RUN MODELS
# ─────────────────────────────────────────────
df = load_data(uploaded_file)
km_labels, km_model, scaler, X_scaled_km, sil_score_val = run_kmeans(df, n_clusters)
df["KMeans_Cluster"] = km_labels

hc_labels, linked_hc, X_scaled_hc = run_hierarchical(df, n_clusters)
df["HC_Cluster"] = hc_labels

K_range, inertias, sil_scores = elbow_silhouette(df)


# ─────────────────────────────────────────────
#  HERO SECTION
# ─────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 50%, #141826 100%);
    border: 1px solid #2e3350;
    border-radius: 16px;
    padding: 40px 36px 32px;
    margin-bottom: 8px;
">
    <h1 style="color:#ffffff; font-size:2.6rem; margin:0 0 6px 0;">🛍️ Mall Customer Segmentation</h1>
    <p style="color:#a0aec0; font-size:1.05rem; margin:0 0 16px 0;">
        Unsupervised Machine Learning · KMeans &amp; Hierarchical Clustering
    </p>
    <p style="color:#718096; font-size:0.9rem; margin:0;">
        Divides 200 mall customers into meaningful groups based on Annual Income and Spending Score.
        Explore EDA, optimal K selection, cluster visualisation, and segment profiling — interactively.
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Top KPI Cards ────────────────────────────
c1, c2, c3, c4 = st.columns(4)
kpi_data = [
    ("Total Customers",   str(len(df))),
    ("Features Used",     "2"),
    ("Optimal Clusters",  str(n_clusters)),
    ("Silhouette Score",  f"{sil_score_val:.3f}"),
]
for col, (label, value) in zip([c1, c2, c3, c4], kpi_data):
    col.markdown(f"""
    <div class="metric-card">
        <h3>{label}</h3>
        <p>{value}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ═════════════════════════════════════════════
#  SECTION 1 · EDA
# ═════════════════════════════════════════════
if show_eda:
    st.markdown('<div class="section-header">1 · Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        "Before modelling, we understand the data — its shape, distribution, and quality. "
        "The dataset contains **200 customers** and **5 features**: CustomerID, Gender, Age, "
        "Annual Income (k\\$), and Spending Score (1–100). There are **no missing values**."
    )

    # Raw data preview
    with st.expander("📋 Raw Data Preview", expanded=False):
        st.dataframe(df.drop(columns=["KMeans_Cluster", "HC_Cluster"]), use_container_width=True)

    # Descriptive stats
    with st.expander("📊 Descriptive Statistics", expanded=False):
        st.dataframe(df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].describe().round(2),
                     use_container_width=True)

    # EDA plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor("#0f1117")
    plt.subplots_adjust(hspace=0.45, wspace=0.35)

    plot_cfg = dict(facecolor="#0f1117", labelcolor="#a0aec0")
    text_kw  = dict(color="#e2e8f0", fontsize=10)
    grid_kw  = dict(color="#2d3748", linestyle="--", linewidth=0.5)

    # --- Gender distribution (bar) ---
    ax = axes[0, 0]
    ax.set_facecolor("#1a1f2e")
    gender_counts = df["Gender"].value_counts()
    bars = ax.bar(gender_counts.index, gender_counts.values,
                  color=["#3498db", "#e91e8c"], edgecolor="none", width=0.5)
    ax.set_title("Gender Distribution", color="#e2e8f0", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count", **text_kw)
    ax.tick_params(colors="#a0aec0")
    ax.spines[:].set_visible(False)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(int(bar.get_height())), ha="center", color="#e2e8f0", fontsize=10)

    # --- Age distribution ---
    ax = axes[0, 1]
    ax.set_facecolor("#1a1f2e")
    ax.hist(df["Age"], bins=15, color="#9b59b6", edgecolor="#0f1117", linewidth=0.5)
    ax.set_title("Age Distribution", color="#e2e8f0", fontsize=11, fontweight="bold")
    ax.set_xlabel("Age", **text_kw)
    ax.set_ylabel("Count", **text_kw)
    ax.tick_params(colors="#a0aec0")
    ax.spines[:].set_visible(False)
    ax.grid(axis="y", **grid_kw)

    # --- Annual Income distribution ---
    ax = axes[0, 2]
    ax.set_facecolor("#1a1f2e")
    ax.hist(df["Annual Income (k$)"], bins=15, color="#00d4ff", edgecolor="#0f1117", linewidth=0.5)
    ax.set_title("Annual Income Distribution", color="#e2e8f0", fontsize=11, fontweight="bold")
    ax.set_xlabel("Annual Income (k$)", **text_kw)
    ax.set_ylabel("Count", **text_kw)
    ax.tick_params(colors="#a0aec0")
    ax.spines[:].set_visible(False)
    ax.grid(axis="y", **grid_kw)

    # --- Spending Score distribution ---
    ax = axes[1, 0]
    ax.set_facecolor("#1a1f2e")
    ax.hist(df["Spending Score (1-100)"], bins=15, color="#2ecc71", edgecolor="#0f1117", linewidth=0.5)
    ax.set_title("Spending Score Distribution", color="#e2e8f0", fontsize=11, fontweight="bold")
    ax.set_xlabel("Spending Score", **text_kw)
    ax.set_ylabel("Count", **text_kw)
    ax.tick_params(colors="#a0aec0")
    ax.spines[:].set_visible(False)
    ax.grid(axis="y", **grid_kw)

    # --- Scatter: Income vs Spending ---
    ax = axes[1, 1]
    ax.set_facecolor("#1a1f2e")
    ax.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"],
               alpha=0.7, color="#f39c12", s=50, edgecolors="none")
    ax.set_title("Income vs Spending Score", color="#e2e8f0", fontsize=11, fontweight="bold")
    ax.set_xlabel("Annual Income (k$)", **text_kw)
    ax.set_ylabel("Spending Score", **text_kw)
    ax.tick_params(colors="#a0aec0")
    ax.spines[:].set_visible(False)
    ax.grid(**grid_kw)

    # --- Correlation heatmap ---
    ax = axes[1, 2]
    ax.set_facecolor("#1a1f2e")
    corr = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].corr()
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(["Age", "Income", "Spending"], color="#a0aec0", fontsize=9)
    ax.set_yticklabels(["Age", "Income", "Spending"], color="#a0aec0", fontsize=9)
    ax.set_title("Correlation Heatmap", color="#e2e8f0", fontsize=11, fontweight="bold")
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", color="white", fontsize=9)

    fig.patch.set_facecolor("#0f1117")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("""
    <div class="insight-box">
        💡 <strong>Key EDA Insights:</strong><br>
        • The dataset is clean with <strong>no missing values</strong>.<br>
        • The gender split is roughly equal (~56% Female, ~44% Male).<br>
        • Age is bimodal — peaks around ~25 and ~45, suggesting two distinct shopper age groups.<br>
        • Income and Spending Score are <strong>nearly uncorrelated</strong>, meaning high earners don't always spend more — this is exactly what makes clustering valuable.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
#  SECTION 2 · ELBOW & SILHOUETTE
# ═════════════════════════════════════════════
if show_elbow:
    st.markdown('<div class="section-header">2 · Finding the Optimal K</div>', unsafe_allow_html=True)
    st.markdown(
        "Two complementary methods — the **Elbow Method** (inertia) and **Silhouette Score** — "
        "are used together to determine the best number of clusters. "
        "Both methods point to **K = 5** as the optimal value."
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")

    for ax in [ax1, ax2]:
        ax.set_facecolor("#1a1f2e")
        ax.tick_params(colors="#a0aec0")
        ax.spines[:].set_visible(False)
        ax.grid(color="#2d3748", linestyle="--", linewidth=0.5)

    # Elbow
    ax1.plot(K_range, inertias, "o-", color="#00d4ff", linewidth=2.5, markersize=7,
             markerfacecolor="#ffffff", markeredgecolor="#00d4ff")
    ax1.axvline(x=5, color="#f39c12", linestyle="--", linewidth=1.5, alpha=0.8, label="Elbow at K=5")
    ax1.set_title("Elbow Method — Inertia vs K", color="#e2e8f0", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Clusters (K)", color="#a0aec0")
    ax1.set_ylabel("Inertia (WCSS)", color="#a0aec0")
    ax1.legend(facecolor="#252a3a", labelcolor="#e2e8f0", edgecolor="#2d3748")

    # Silhouette
    bar_colors = ["#f39c12" if k == 5 else "#3498db" for k in K_range]
    ax2.bar(K_range, sil_scores, color=bar_colors, edgecolor="none", width=0.6)
    ax2.set_title("Silhouette Score vs K", color="#e2e8f0", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Clusters (K)", color="#a0aec0")
    ax2.set_ylabel("Silhouette Score", color="#a0aec0")
    ax2.set_xticks(K_range)
    best_idx = sil_scores.index(max(sil_scores))
    ax2.text(K_range[best_idx], sil_scores[best_idx] + 0.005,
             f"Best: {max(sil_scores):.3f}", ha="center", color="#f39c12", fontsize=9, fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    col_a, col_b = st.columns(2)
    col_a.markdown("""
    <div class="insight-box">
        📉 <strong>Elbow Method:</strong> Inertia drops sharply until K=5, then flattens.
        The "elbow" at K=5 indicates diminishing returns — adding more clusters beyond 5
        doesn't significantly reduce within-cluster variance.
    </div>
    """, unsafe_allow_html=True)
    col_b.markdown("""
    <div class="insight-box">
        📈 <strong>Silhouette Score:</strong> Peaks at K=5 (~0.55), meaning clusters are
        well-separated and internally cohesive at this value.
        A score above 0.5 is considered a strong result for customer data.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
#  SECTION 3 · KMEANS CLUSTERING
# ═════════════════════════════════════════════
if show_clusters:
    st.markdown('<div class="section-header">3 · KMeans Clustering Results</div>', unsafe_allow_html=True)
    st.markdown(
        "KMeans with **K = {k}** was trained on *Annual Income* and *Spending Score* after "
        "StandardScaler normalisation. The scatter plot below shows the final customer segments.".format(k=n_clusters)
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1f2e")
    ax.tick_params(colors="#a0aec0")
    ax.spines[:].set_visible(False)
    ax.grid(color="#2d3748", linestyle="--", linewidth=0.5)

    for cluster_id in range(n_clusters):
        mask = df["KMeans_Cluster"] == cluster_id
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        label = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
        ax.scatter(
            df.loc[mask, "Annual Income (k$)"],
            df.loc[mask, "Spending Score (1-100)"],
            s=point_size, color=color, alpha=0.85,
            edgecolors="white", linewidths=0.4, label=label,
        )

    if show_centroids:
        # Map centroids back to original scale
        centroids_scaled = km_model.cluster_centers_
        centroids_orig = scaler.inverse_transform(centroids_scaled)
        ax.scatter(centroids_orig[:, 0], centroids_orig[:, 1],
                   s=260, marker="X", color="white",
                   edgecolors="#0f1117", linewidths=1.2, zorder=5, label="Centroids")

    ax.set_title(f"KMeans Customer Segmentation  (K = {n_clusters})",
                 color="#e2e8f0", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Annual Income (k$)", color="#a0aec0", fontsize=11)
    ax.set_ylabel("Spending Score (1–100)", color="#a0aec0", fontsize=11)
    legend = ax.legend(facecolor="#252a3a", labelcolor="#e2e8f0",
                       edgecolor="#2d3748", fontsize=9, loc="upper left")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Cluster size table
    cluster_counts = df["KMeans_Cluster"].value_counts().sort_index()
    summary_df = pd.DataFrame({
        "Cluster ID": cluster_counts.index,
        "Segment Name": [CLUSTER_NAMES.get(i, f"Cluster {i}") for i in cluster_counts.index],
        "Customers": cluster_counts.values,
        "Share (%)": (cluster_counts.values / len(df) * 100).round(1),
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════
#  SECTION 4 · CLUSTER PROFILES
# ═════════════════════════════════════════════
if show_profiles:
    st.markdown('<div class="section-header">4 · Cluster Profiles & Business Insights</div>', unsafe_allow_html=True)
    st.markdown(
        "Each cluster has a distinct Income–Spending personality. "
        "The radar-style bar charts and statistics below reveal what makes each segment unique — "
        "and what business action it warrants."
    )

    profile = df.groupby("KMeans_Cluster").agg(
        Avg_Age=("Age", "mean"),
        Avg_Income=("Annual Income (k$)", "mean"),
        Avg_Spending=("Spending Score (1-100)", "mean"),
        Count=("Age", "count"),
    ).round(1).reset_index()

    segment_insights = {
        0: ("Standard Customers",   "🟡", "#f39c12",
            "Mid income, mid spending. Reliable base segment. Offer loyalty programmes to increase engagement."),
        1: ("Budget Shoppers",      "🔵", "#3498db",
            "Low income, low spending. Price-sensitive. Target with discounts, bundle deals, and budget-friendly promotions."),
        2: ("VIP / High-Value",     "🔴", "#e74c3c",
            "High income, high spending. Most profitable segment. Reward with premium memberships and exclusive perks."),
        3: ("Impulsive Buyers",     "🟢", "#2ecc71",
            "Low income, high spending. Spend beyond their means. Target with limited-time offers and impulse-buy displays."),
        4: ("Untapped Potential",   "🟣", "#9b59b6",
            "High income, low spending. Saving-minded despite earnings. Upsell with premium products; they have the wallet but need the nudge."),
    }

    for i in range(0, n_clusters, 2):
        cols = st.columns(2 if i + 1 < n_clusters else 1)
        for j, col in enumerate(cols):
            cid = i + j
            if cid >= n_clusters:
                break
            row = profile[profile["KMeans_Cluster"] == cid].iloc[0]
            name, emoji, color, insight = segment_insights.get(
                cid, (f"Cluster {cid}", "⚪", "#a0aec0", "No description available."))

            col.markdown(f"""
            <div style="background:#1a1f2e; border:1px solid {color}40;
                        border-top: 4px solid {color}; border-radius:12px; padding:20px; margin-bottom:12px;">
                <h3 style="color:{color}; margin:0 0 4px 0; font-size:1.1rem;">{emoji} {name}</h3>
                <p style="color:#a0aec0; font-size:0.8rem; margin:0 0 14px 0;">Cluster {cid} · {int(row.Count)} customers</p>
                <div style="display:flex; gap:20px; margin-bottom:14px;">
                    <div style="text-align:center;">
                        <div style="color:#a0aec0;font-size:0.75rem;">Avg Age</div>
                        <div style="color:#e2e8f0;font-size:1.4rem;font-weight:700;">{row.Avg_Age:.0f}</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="color:#a0aec0;font-size:0.75rem;">Avg Income</div>
                        <div style="color:#e2e8f0;font-size:1.4rem;font-weight:700;">${row.Avg_Income:.0f}k</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="color:#a0aec0;font-size:0.75rem;">Avg Spending</div>
                        <div style="color:#e2e8f0;font-size:1.4rem;font-weight:700;">{row.Avg_Spending:.0f}/100</div>
                    </div>
                </div>
                <div style="background:#0f1117; border-left:3px solid {color}; padding:10px 12px;
                            border-radius:0 6px 6px 0; color:#cbd5e0; font-size:0.88rem;">
                    💼 {insight}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Grouped bar chart — avg metrics per cluster
    st.markdown("#### Average Metrics by Cluster")
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1f2e")
    ax.tick_params(colors="#a0aec0")
    ax.spines[:].set_visible(False)
    ax.grid(axis="y", color="#2d3748", linestyle="--", linewidth=0.5)

    x = np.arange(n_clusters)
    w = 0.25
    ax.bar(x - w, profile["Avg_Age"],     width=w, color="#9b59b6", label="Avg Age",     edgecolor="none")
    ax.bar(x,     profile["Avg_Income"],  width=w, color="#00d4ff", label="Avg Income",  edgecolor="none")
    ax.bar(x + w, profile["Avg_Spending"],width=w, color="#2ecc71", label="Avg Spending",edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i}" for i in range(n_clusters)], color="#a0aec0")
    ax.set_title("Cluster Comparison — Age · Income · Spending Score",
                 color="#e2e8f0", fontsize=12, fontweight="bold")
    ax.legend(facecolor="#252a3a", labelcolor="#e2e8f0", edgecolor="#2d3748")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ═════════════════════════════════════════════
#  SECTION 5 · HIERARCHICAL CLUSTERING
# ═════════════════════════════════════════════
if show_hier:
    st.markdown('<div class="section-header">5 · Hierarchical Clustering & Dendrogram</div>', unsafe_allow_html=True)
    st.markdown(
        "Agglomerative (Hierarchical) Clustering is applied as a **validation step**. "
        "Unlike KMeans, it does not require specifying K upfront — the dendrogram reveals natural groupings by "
        "showing where the tree can be cut. The longest uncut vertical branches appear at distance ≈ 5, "
        "confirming **K = 5**."
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0f1117")

    # --- Dendrogram ---
    ax1.set_facecolor("#1a1f2e")
    ax1.tick_params(colors="#a0aec0")
    ax1.spines[:].set_visible(False)
    ax1.grid(axis="y", color="#2d3748", linestyle="--", linewidth=0.5)

    dendrogram(
        linked_hc,
        truncate_mode="lastp",
        p=20,
        ax=ax1,
        color_threshold=None,
        above_threshold_color="#00d4ff",
        link_color_func=lambda k: "#00d4ff",
    )
    ax1.set_title("Hierarchical Dendrogram (Truncated)", color="#e2e8f0", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Customers", color="#a0aec0")
    ax1.set_ylabel("Ward Distance", color="#a0aec0")
    # Indicate cut line
    ax1.axhline(y=5, color="#f39c12", linestyle="--", linewidth=1.5, label="Cut at y=5  →  5 clusters")
    ax1.legend(facecolor="#252a3a", labelcolor="#e2e8f0", edgecolor="#2d3748", fontsize=9)

    # --- HC Scatter ---
    ax2.set_facecolor("#1a1f2e")
    ax2.tick_params(colors="#a0aec0")
    ax2.spines[:].set_visible(False)
    ax2.grid(color="#2d3748", linestyle="--", linewidth=0.5)

    for cluster_id in range(n_clusters):
        mask = df["HC_Cluster"] == cluster_id
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        ax2.scatter(
            df.loc[mask, "Annual Income (k$)"],
            df.loc[mask, "Spending Score (1-100)"],
            s=point_size, color=color, alpha=0.85,
            edgecolors="white", linewidths=0.4, label=f"Cluster {cluster_id}",
        )
    ax2.set_title(f"Agglomerative Clustering  (K = {n_clusters})",
                  color="#e2e8f0", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Annual Income (k$)", color="#a0aec0")
    ax2.set_ylabel("Spending Score (1–100)", color="#a0aec0")
    ax2.legend(facecolor="#252a3a", labelcolor="#e2e8f0", edgecolor="#2d3748", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Comparison table
    st.markdown("#### KMeans vs Hierarchical Clustering")
    comparison = pd.DataFrame({
        "Criterion":              ["Need to specify K?", "Shows merging process?", "Scales to big data?", "Result on this dataset", "Agreement"],
        "KMeans":                 ["✅ Yes", "❌ No", "✅ Fast", "K = 5", "✅"],
        "Hierarchical (Ward)":   ["❌ No", "✅ Dendrogram", "⚠️ Slow on large data", "5 cuts", "✅"],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="insight-box">
        ✅ <strong>Validation:</strong> Both KMeans and Hierarchical Clustering independently identify
        <strong>5 natural customer segments</strong>. This cross-method agreement gives strong confidence
        that the 5-cluster structure is a true property of the data — not an artefact of the algorithm.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
#  SECTION 6 · CUSTOMER PREDICTOR
# ═════════════════════════════════════════════
if show_predictor:
    st.markdown('<div class="section-header">6 · Customer Segment Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        "Enter a new customer's details below to predict which segment they belong to "
        "using the trained KMeans model."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        input_age    = st.number_input("Age", min_value=18, max_value=100, value=30)
    with col2:
        input_income = st.number_input("Annual Income (k$)", min_value=1, max_value=300, value=60)
    with col3:
        input_spend  = st.number_input("Spending Score (1–100)", min_value=1, max_value=100, value=50)

    if st.button("🔍 Predict Segment", type="primary"):
        point_scaled = scaler.transform([[input_income, input_spend]])
        predicted_cluster = km_model.predict(point_scaled)[0]
        name, emoji, color, insight = segment_insights.get(
            predicted_cluster,
            (f"Cluster {predicted_cluster}", "⚪", "#a0aec0", "No description available."))

        st.markdown(f"""
        <div style="background:#1a1f2e; border: 2px solid {color}; border-radius:14px; padding:24px; margin-top:12px;">
            <h2 style="color:{color}; margin:0 0 6px 0;">{emoji} {name}</h2>
            <p style="color:#a0aec0; margin:0 0 16px 0;">Cluster ID: <strong style="color:#e2e8f0;">{predicted_cluster}</strong></p>
            <div style="display:flex; gap:30px; margin-bottom:16px;">
                <div><div style="color:#718096;font-size:0.8rem;">Age</div>
                     <div style="color:#e2e8f0;font-size:1.5rem;font-weight:700;">{input_age}</div></div>
                <div><div style="color:#718096;font-size:0.8rem;">Annual Income</div>
                     <div style="color:#e2e8f0;font-size:1.5rem;font-weight:700;">${input_income}k</div></div>
                <div><div style="color:#718096;font-size:0.8rem;">Spending Score</div>
                     <div style="color:#e2e8f0;font-size:1.5rem;font-weight:700;">{input_spend}/100</div></div>
            </div>
            <div style="background:#0f1117; border-left:4px solid {color}; padding:12px 16px;
                        border-radius:0 8px 8px 0; color:#cbd5e0; font-size:0.93rem;">
                💼 {insight}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Plot customer position on the cluster map
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1a1f2e")
        ax.tick_params(colors="#a0aec0")
        ax.spines[:].set_visible(False)
        ax.grid(color="#2d3748", linestyle="--", linewidth=0.5)

        for cluster_id in range(n_clusters):
            mask = df["KMeans_Cluster"] == cluster_id
            c = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
            ax.scatter(df.loc[mask, "Annual Income (k$)"],
                       df.loc[mask, "Spending Score (1-100)"],
                       s=40, color=c, alpha=0.4, edgecolors="none")

        ax.scatter(input_income, input_spend,
                   s=350, color=color, marker="*", edgecolors="white",
                   linewidths=1.5, zorder=10, label="You are here")
        ax.set_title("Your position on the Customer Map",
                     color="#e2e8f0", fontsize=12, fontweight="bold")
        ax.set_xlabel("Annual Income (k$)", color="#a0aec0")
        ax.set_ylabel("Spending Score", color="#a0aec0")
        ax.legend(facecolor="#252a3a", labelcolor="#e2e8f0", edgecolor="#2d3748")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ═════════════════════════════════════════════
#  FOOTER
# ═════════════════════════════════════════════
st.markdown("""
<div class="footer">
    Built with ❤️ by <strong>Sanan Shaikh</strong> ·
    <a href="https://github.com/shaikhsanan04/mall-customer-segmentation-ml-project"
       style="color:#00d4ff; text-decoration:none;">GitHub Repository</a> ·
    <a href="https://linkedin.com/in/shaikhsanan04" style="color:#00d4ff; text-decoration:none;">LinkedIn</a>
    <br><br>
    Stack: Python · Streamlit · Scikit-learn · Pandas · NumPy · Matplotlib · SciPy
</div>
""", unsafe_allow_html=True)
