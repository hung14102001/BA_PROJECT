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
import superstore_analysis


def main():
    sns.set_theme(style="whitegrid")
    
    # Load data
    inp = Path("./data/cleaned_data.csv")
    if not inp.exists():
        print(f"[ERR] Không tìm thấy file: {inp}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path("./analysis_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    # Detect columns
    sales_col = superstore_analysis.find_col(df.columns, ["Sales","sales","Revenue","Amount"])
    date_col = superstore_analysis.find_col(df.columns, ["Order Date","OrderDate","Date","Ship Date"])
    cust_col = superstore_analysis.find_col(df.columns, ["Customer ID","CustomerID","Customer","customer_id"])
    profit_col = superstore_analysis.find_col(df.columns, ["Profit","profit"])
    qty_col = superstore_analysis.find_col(df.columns, ["Quantity","Qty","Order Quantity"])
    disc_col = superstore_analysis.find_col(df.columns, ["Discount","discount","Disc","%Discount"])
    ship_col = superstore_analysis.find_col(df.columns, ["Ship Date","ShipDate"])
    prod_col = superstore_analysis.find_col(df.columns, ["Product Name","Product","Product ID"])


    # Build RFM features
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    today = df[date_col].max() + timedelta(days=1)
    rfm = df.groupby(cust_col).agg(
            last_date=(date_col, "max"),       # Ngày mua gần nhất
            frequency=(cust_col, "size"),      # Số lượng giao dịch
            monetary=(sales_col, "sum"),       # Tổng doanh thu
            aov=(sales_col, "mean")            # Giá trị trung bình mỗi đơn hàng (Average Order Value)
    ).reset_index()
    rfm["recency_days"] = (today - rfm["last_date"]).dt.days

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

    # === ELBOW + SILHOUETTE ANALYSIS ===
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
    superstore_analysis.safe_save_fig(fig, out_dir / "kmeans_elbow_silhouette.png")

    # Select best k
    best_idx = np.argmax(sil_list)
    k_best = range_n_clusters[best_idx]
    print(f"\n→ Chọn k tối ưu theo Silhouette: k={k_best}")

    # === FIT KMEANS WITH BEST K ===
    kmeans = KMeans(n_clusters=k_best, n_init=50, max_iter=500, random_state=42)
    labels = kmeans.fit_predict(X)
    rfm["Cluster"] = labels
    
    sil_best = silhouette_score(X, labels)
    print(f"✓ Silhouette score (k={k_best}): {sil_best:.4f}")


    # === KPI CLUSTER ===
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


    # === Hiệu quả tài chính + quy mô khách hàng theo cụm
    fig, ax1 = plt.subplots(figsize=(8,5))

    x = np.arange(len(kpi["Cluster"]))
    width = 0.35

    # --- Cột: Total Sales & Total Profit ---
    ax1.bar(x - width/2, kpi["total_sales"], width, label="Tổng doanh thu", color="#ff9966")
    ax1.bar(x + width/2, kpi["total_profit"], width, label="Tổng lợi nhuận", color="#66b3ff")

    ax1.set_xlabel("Cụm khách hàng", fontsize=11)
    ax1.set_ylabel("Giá trị ($)", fontsize=11)
    ax1.set_title("Hiệu quả tài chính & số lượng khách hàng theo cụm", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(kpi["Cluster"].astype(str))

    # Gắn nhãn giá trị trên cột
    for i, v in enumerate(kpi["total_sales"]):
        ax1.text(i - width/2, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(kpi["total_profit"]):
        ax1.text(i + width/2, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)

    # --- Trục phụ: số lượng khách hàng ---
    ax2 = ax1.twinx()
    cnt = rfm["Cluster"].value_counts().sort_index()
    ax1.ticklabel_format(style='plain', axis='y')
    ax2.plot(x, cnt.values, color="green", marker="o", linewidth=2.5, label="Số khách hàng")
    ax2.set_ylabel("Số khách hàng", color="green", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="green")

    # Gắn nhãn số khách hàng
    for i, v in enumerate(cnt.values):
        ax2.text(i, v + max(cnt.values)*0.02, f"{v:,}", ha="center", va="bottom", color="green", fontsize=9)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.tight_layout()
    superstore_analysis.safe_save_fig(fig, out_dir / "kpi_financial_customer_by_cluster.png")


    # Biểu đồ hành vi khách hàng (AOV, Frequency, Recency)
    fig, ax2 = plt.subplots(figsize=(8,5))

    metrics = ["avg_order_value", "avg_frequency", "avg_recency_days"]
    labels = [
        "Giá trị TB/đơn ($)",
        "Tần suất mua TB (lần)",
        "Số ngày từ lần mua gần nhất (ngày)"
    ]
    colors = ["#66c2a5", "#fc8d62", "#8da0cb"]

    for i, m in enumerate(metrics):
        if m in kpi.columns:
            y = kpi[m]
            ax2.plot(
                kpi["Cluster"].astype(str),
                y,
                marker="o",
                label=labels[i],
                color=colors[i],
                linewidth=2
            )
            for j, val in enumerate(y):
                ax2.text(
                    j,
                    val + (max(y) * 0.02), 
                    f"{val:,.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=colors[i]
                )

    ax2.set_xlabel("Cụm khách hàng", fontsize=11)
    ax2.set_ylabel("Giá trị trung bình", fontsize=11)
    ax2.set_title("Hành vi khách hàng theo cụm", fontsize=14, fontweight="bold")
    ax2.legend()
    plt.tight_layout()
    superstore_analysis.safe_save_fig(fig, out_dir / "kpi_behavior_by_cluster.png")

    # === PAIRPLOT ===

    sns.set_style(style="ticks")
    pp = sns.pairplot(rfm, hue="Cluster", vars=feat_cols, palette="Set2", corner=True, plot_kws={'alpha':0.6})
    pp.fig.suptitle("Quan hệ cụm - RFM Features", y=1.02, fontsize=14, fontweight='bold')
    superstore_analysis.safe_save_fig(pp.fig, out_dir / f"pairplot_k{k_best}.png")


    # === PHÂN TÍCH HÀNH VI MUA HÀNG THEO THỜI GIAN ===

    if date_col in df.columns:
        # Chuẩn hóa ngày tháng
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["Month"] = df[date_col].dt.month

        # Gắn Cluster vào từng giao dịch
        df = df.merge(rfm[[cust_col, "Cluster"]], on=cust_col, how="left")

        # === 1. Phân tích theo tháng
        monthly_sales = (
            df.groupby(["Cluster", "Month"])[sales_col]
            .sum()
            .reset_index()
            .sort_values(["Cluster", "Month"])
        )

        fig, ax = plt.subplots(figsize=(9,5))
        sns.lineplot(
            data=monthly_sales,
            x="Month",
            y=sales_col,
            hue="Cluster",
            marker="o",
            palette="Set2",
            ax=ax
        )
        ax.set_title("Doanh thu theo tháng - từng cụm khách hàng", fontsize=14, fontweight="bold")
        ax.set_xlabel("Tháng", fontsize=11)
        ax.set_ylabel("Tổng doanh thu ($)", fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        superstore_analysis.safe_save_fig(fig, out_dir / "sales_by_month_by_cluster.png")



    # === TOP PRODUCTS BY CLUSTER ===
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
                superstore_analysis.safe_save_fig(fig, out_dir / f"top_products_cluster_{c}.png")

if __name__ == "__main__":
    main()
