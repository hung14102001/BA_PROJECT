from datetime import timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def find_col(cols, candidates):
    for cand in candidates:
        for c in cols:
            if str(c).strip().lower() == cand.strip().lower():
                return c
    return None

def safe_save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)

def main():
    sns.set_theme(style="whitegrid")
    
    # Load data
    inp = Path("./data/cleaned_data.csv")
    if not inp.exists():
        print(f"[ERR] Không tìm thấy file: {inp}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path("./analysis_outputs2")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    # Detect columns
    sales_col = find_col(df.columns, ["Sales","sales","Revenue","Amount"])
    date_col = find_col(df.columns, ["Order Date","OrderDate","Date","Ship Date"])
    cust_col = find_col(df.columns, ["Customer ID","CustomerID","Customer","customer_id"])
    profit_col = find_col(df.columns, ["Profit","profit"])
    qty_col = find_col(df.columns, ["Quantity","Qty","Order Quantity"])
    disc_col = find_col(df.columns, ["Discount","discount","Disc","%Discount"])
    ship_col = find_col(df.columns, ["Ship Date","ShipDate"])
    prod_col = find_col(df.columns, ["Product Name","Product","Product ID"])

    if any(c is None for c in [sales_col, date_col, cust_col]):
        print("[ERR] Thiếu cột Sales / Order Date / Customer ID trong CSV.", file=sys.stderr)
        sys.exit(2)

    # Build RFM features
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    today = df[date_col].max() + timedelta(days=1)
    rfm = df.groupby(cust_col).agg(
        last_date=(date_col, "max"),
        frequency=(cust_col, "size"),
        monetary=(sales_col, "sum"),
        aov=(sales_col, "mean")
    ).reset_index()
    rfm["recency_days"] = (today - rfm["last_date"]).dt.days

    # Merge optional aggregates
    if profit_col:
        prof_by_cust = df.groupby(cust_col)[profit_col].sum().rename("total_profit")
        rfm = rfm.merge(prof_by_cust, on=cust_col, how="left")
    
    if disc_col:
        disc_by_cust = df.groupby(cust_col)[disc_col].mean().rename("avg_discount")
        rfm = rfm.merge(disc_by_cust, on=cust_col, how="left")
    
    if "ship_delay_days" in df.columns:
        ship_by_cust = df.groupby(cust_col)["ship_delay_days"].mean().rename("avg_ship_delay_days")
        rfm = rfm.merge(ship_by_cust, on=cust_col, how="left")

    # Feature matrix for clustering
    feat_cols = ["recency_days","frequency","monetary","aov"]
    X_raw = rfm[feat_cols].fillna(0.0)

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # === 1) ELBOW + SILHOUETTE ANALYSIS ===
    print("\n=== Analyzing optimal k ===")
    range_n_clusters = [2,3,4,5,6,7,8]
    ssd_list = []
    sil_list = []

    for k in range_n_clusters:
        km = KMeans(n_clusters=k, max_iter=300, n_init=30, random_state=42)
        km.fit(X)
        inertia = float(km.inertia_)
        ssd_list.append(inertia)
        score = float(silhouette_score(X, km.labels_))
        sil_list.append(score)
        print(f"k={k}: SSD={inertia:.2f}, Silhouette={score:.4f}")

    # Plot Elbow + Silhouette
    fig, ax1 = plt.subplots(figsize=(9,6))
    ax1.plot(range_n_clusters, ssd_list, marker="o", label="SSD (inertia)", color="tab:blue")
    ax1.set_xlabel("Số cụm (k)", fontsize=11)
    ax1.set_ylabel("SSD (inertia)", fontsize=11, color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(range_n_clusters, sil_list, marker="s", linestyle="--", color="tab:orange", label="Silhouette")
    ax2.set_ylabel("Silhouette Score", fontsize=11, color="tab:orange")
    ax2.tick_params(axis='y', labelcolor="tab:orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_title("KMeans: Elbow & Silhouette Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    safe_save_fig(fig, out_dir / "kmeans_elbow_silhouette.png")

    # Select best k
    best_idx = np.argmax(sil_list)
    k_best = range_n_clusters[best_idx]
    print(f"\n→ Chọn k tối ưu theo Silhouette: k={k_best}")

    # === 2) FIT KMEANS WITH BEST K ===
    kmeans = KMeans(n_clusters=k_best, n_init=50, max_iter=500, random_state=42)
    labels = kmeans.fit_predict(X)
    rfm["Cluster"] = labels
    
    sil_best = silhouette_score(X, labels)
    print(f"✓ Silhouette score (k={k_best}): {sil_best:.4f}")

    # === 3) CLUSTER DISTRIBUTION ===
    cnt = rfm["Cluster"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar(cnt.index.astype(str), cnt.values, color='skyblue', edgecolor='navy')
    ax.set_xlabel("Cluster", fontsize=11)
    ax.set_ylabel("Số khách hàng", fontsize=11)
    ax.set_title("Phân bố số khách hàng theo cụm", fontsize=14, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{int(height):,}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    safe_save_fig(fig, out_dir / "cluster_distribution.png")
    print(f"✓ Saved: {out_dir / 'cluster_distribution.png'}")

    # === 4) KPI BY CLUSTER ===
    kpi = rfm.groupby("Cluster").agg(
        customers=(cust_col, "nunique"),
        orders=("frequency", "sum"),
        total_sales=("monetary", "sum"),
        avg_order_value=("aov", "mean"),
        avg_frequency=("frequency","mean"),
        avg_recency_days=("recency_days","mean")
    ).reset_index()

    if "avg_discount" in rfm.columns:
        kpi = kpi.merge(rfm.groupby("Cluster")["avg_discount"].mean().reset_index(), on="Cluster")
    if "total_profit" in rfm.columns:
        kpi = kpi.merge(rfm.groupby("Cluster")["total_profit"].sum().reset_index(), on="Cluster")

    # Plot KPI metrics
    metrics = ["total_sales","avg_order_value","avg_frequency","avg_recency_days"]
    if "total_profit" in kpi.columns:
        metrics.append("total_profit")
    
    for m in metrics:
        if m in kpi.columns:
            fig, ax = plt.subplots(figsize=(8,5))
            bars = ax.bar(kpi["Cluster"].astype(str), kpi[m].values, color='coral', edgecolor='darkred')
            ax.set_xlabel("Cluster", fontsize=11)
            ax.set_ylabel(m.replace("_"," ").title(), fontsize=11)
            ax.set_title(f"KPI theo cụm: {m.replace('_',' ').title()}", fontsize=14, fontweight='bold')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height,
                        f'{height:,.0f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            safe_save_fig(fig, out_dir / f"kpi_{m}_by_cluster.png")
            print(f"✓ Saved: {out_dir / f'kpi_{m}_by_cluster.png'}")

    # === 5) RADAR CHART - CLUSTER CENTERS ===
    centers_unscaled = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=feat_cols
    )
    
    # Normalize for radar chart
    norm = (centers_unscaled - centers_unscaled.min()) / (centers_unscaled.max() - centers_unscaled.min() + 1e-9)
    
    labels_radar = feat_cols
    angles = np.linspace(0, 2*np.pi, len(labels_radar), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(111, polar=True)
    
    colors = plt.cm.Set2(range(len(norm)))
    for idx, row in norm.iterrows():
        vals = row.values.tolist()
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=2, label=f'Cluster {idx}', color=colors[idx])
        ax.fill(angles, vals, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_radar, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Tâm cụm (chuẩn hóa) - Radar Chart", fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    safe_save_fig(fig, out_dir / "centers_radar_chart.png")
    print(f"✓ Saved: {out_dir / 'centers_radar_chart.png'}")

    # === 6) PAIRPLOT ===
    try:
        sns.set(style="ticks")
        pp = sns.pairplot(rfm, hue="Cluster", vars=feat_cols, palette="Set2", corner=True, plot_kws={'alpha':0.6})
        pp.fig.suptitle("Quan hệ cụm - RFM Features", y=1.02, fontsize=14, fontweight='bold')
        safe_save_fig(pp.fig, out_dir / f"pairplot_k{k_best}.png")
        print(f"✓ Saved: {out_dir / f'pairplot_k{k_best}.png'}")
    except Exception as e:
        print(f"[WARN] Không thể vẽ pairplot: {e}")

    # === 7) TOP PRODUCTS BY CLUSTER ===
    if prod_col and prod_col in df.columns:
        trans = df[[cust_col, prod_col, sales_col]].copy()
        trans = trans.merge(rfm[[cust_col, "Cluster"]], on=cust_col, how="left")
        
        for c in sorted(rfm["Cluster"].unique()):
            sub = trans[trans["Cluster"]==c]
            top_prod = sub.groupby(prod_col)[sales_col].sum().sort_values(ascending=False).head(10)
            
            if len(top_prod) > 0:
                fig, ax = plt.subplots(figsize=(10,6))
                ax.barh(range(len(top_prod)), top_prod.values, color='lightgreen', edgecolor='darkgreen')
                ax.set_yticks(range(len(top_prod)))
                ax.set_yticklabels([str(p)[:50] for p in top_prod.index], fontsize=9)
                ax.set_xlabel("Total Sales ($)", fontsize=11)
                ax.set_title(f"Top 10 sản phẩm theo Cluster {c}", fontsize=14, fontweight='bold')
                ax.invert_yaxis()
                
                for i, v in enumerate(top_prod.values):
                    ax.text(v, i, f' ${v:,.0f}', va='center', fontsize=9)
                
                plt.tight_layout()
                safe_save_fig(fig, out_dir / f"top_products_cluster_{c}.png")
                print(f"✓ Saved: {out_dir / f'top_products_cluster_{c}.png'}")

    print(f"\n{'='*50}")
    print(f"✓ Hoàn thành! Tất cả visualization đã được lưu vào: {out_dir}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
