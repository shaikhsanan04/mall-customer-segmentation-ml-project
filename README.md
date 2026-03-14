# 🛍️ Mall Customer Segmentation
### Unsupervised Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.14-blue)
![sklearn](https://img.shields.io/badge/scikit--learn-clustering-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Project Overview

This project applies **unsupervised machine learning** to segment mall customers
into distinct groups based on their **Annual Income** and **Spending Score**.
The goal is to help businesses understand their customer base and build
targeted marketing strategies for each segment.

This is a beginner-friendly end-to-end ML project covering the full pipeline —
from raw data exploration to actionable business recommendations.

---

## 📂 Dataset

- **Source:** [Kaggle — Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- **Rows:** 200 customers
- **Features:** CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)

---

## 🛠️ Tech Stack

- Python 3.14
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (KMeans, AgglomerativeClustering, StandardScaler, silhouette_score)
- SciPy (dendrogram, linkage)
- JupyterLab

---

## 📊 Project Workflow

1. **Data Loading & Exploration** — shape, info, describe, null/duplicate checks
2. **Distribution Analysis** — histograms with KDE for Age, Income, Spending
3. **Pre-clustering Scatter Plot** — Income vs Spending to visually identify groups
4. **Feature Scaling** — StandardScaler to normalize features for distance-based clustering
5. **Elbow Method** — finding optimal K by plotting inertia for K=1–10
6. **Silhouette Score** — confirming K by measuring cluster quality for K=2–10
7. **KMeans Clustering** — training final model with K=5, assigning cluster labels
8. **Cluster Visualization** — scatter plot with 5 color-coded segments and centroids
9. **Cluster Interpretation** — mean values per cluster for business profiling
10. **Hierarchical Clustering** — dendrogram using Ward's method to independently validate K=5

---

## 🎯 Results — 5 Customer Segments

| Cluster | Income | Spending | Segment | Business Priority |
|---------|--------|----------|---------|-------------------|
| 💎 VIP | High | High | Premium Customers | Retain & Reward |
| 🎯 Impulsive | Low | High | Discount Hunters | Promotions & Deals |
| 😴 Untapped | High | Low | High Income Savers | Convert & Engage |
| 😐 Budget | Low | Low | Careful Spenders | Retain with Value |
| 🧑 Standard | Medium | Medium | Average Customers | Upsell Opportunity |

**Silhouette Score: 0.5547** (Reasonable cluster quality for real-world data)  
Both Elbow Method and Silhouette Score independently confirmed **K=5**.

---

## 💼 Key Business Insight

> Income and spending are **not correlated**.  
> High-income customers don't always spend more — the **Untapped segment**
> (high income, low spending) represents the biggest missed revenue opportunity.

---

## 🚀 How to Run
```bash
# 1. Clone the repo
git clone https://github.com/shaikhsanan04/mall-customer-segmentation-ml-project.git

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyterlab

# 3. Launch JupyterLab
jupyter lab

# 4. Open Mall_Customer_Segmentation.ipynb and run all cells
```

---

## 📚 What I Learned

- Feature scaling and why it matters for distance-based algorithms
- How to use the Elbow Method and Silhouette Score to find optimal K
- How KMeans assigns clusters using centroids and Euclidean distance
- How to read and interpret a Dendrogram
- How to translate ML results into real business recommendations

---

## 🔗 Connect

**GitHub:** [shaikhsanan04](https://github.com/shaikhsanan04)
**Live App** [https://mall-customer-segmentation-ml.streamlit.app/]
