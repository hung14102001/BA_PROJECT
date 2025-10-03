import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Optional: seaborn for prettier plots
try:
    import seaborn as sns
except Exception:
    sns = None

def find_col(cols, candidates):
    for cand in candidates:
        for c in cols:
            if str(c).strip().lower() == cand.strip().lower():
                return c
    return None

def main():

    inp = Path("./data/sample_age.csv")
    if not inp.exists():
        print(f"[ERR] Không tìm thấy file: {inp}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path("./analysis_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(inp, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    # Detect columns
    sales_col = find_col(df.columns, ["Sales","sales","Revenue","Amount"])
    date_col  = find_col(df.columns, ["Order Date","OrderDate","Date","Ship Date"])
    cust_col  = find_col(df.columns, ["Customer ID","CustomerID","Customer","customer_id"])
    profit_col= find_col(df.columns, ["Profit","profit"])
    qty_col   = find_col(df.columns, ["Quantity","Qty","Order Quantity"])
    disc_col  = find_col(df.columns, ["Discount","discount","Disc","%Discount"])
    ship_col  = find_col(df.columns, ["Ship Date","ShipDate"])
    prod_col  = find_col(df.columns, ["Product Name","Product","Product ID"])

    if any(c is None for c in [sales_col, date_col, cust_col]):
        print("[ERR] Thiếu cột Sales / Order Date / Customer ID trong CSV.", file=sys.stderr)
        sys.exit(2)

    # Clean
    df[sales_col] = pd.to_numeric(df[sales_col].astype(str).str.replace(r'[\\$,]','', regex=True), errors='coerce').fillna(0.0)
    df[date_col]  = pd.to_datetime(df[date_col], errors='coerce')
    if profit_col:
        df[profit_col] = pd.to_numeric(df[profit_col].astype(str).str.replace(r'[\\$,]','', regex=True), errors='coerce').fillna(0.0)
    if disc_col:
        d = pd.to_numeric(df[disc_col].astype(str).str.replace('%','', regex=False), errors='coerce')
        if d.max() and d.max() > 1.5: d = d/100.0
        df[disc_col] = d.fillna(0.0)
    if ship_col:
        df[ship_col] = pd.to_datetime(df[ship_col], errors='coerce')
        df["ship_delay_days"] = (df[ship_col] - df[date_col]).dt.days

    df = df[df[date_col].notna()].copy()

    # Build RFM-like features per customer
    today = df[date_col].max() + timedelta(days=1)
    rfm = df.groupby(cust_col).agg(
        last_date=(date_col, "max"),
        frequency=(cust_col, "size"),
        monetary=(sales_col, "sum"),
        aov=(sales_col, "mean")
    ).reset_index()
    rfm["recency_days"] = (today - rfm["last_date"]).dt.days

    # Merge optional per-customer aggregates
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

    # KMeans
        # ---- Tìm k tối ưu: so sánh SSD (Elbow) + Silhouette ----


    range_n_clusters = [1,2,3,4,5,6,7,8]
    rows = []
    ssd = []
    sil = []

    for k in range_n_clusters:
        km = KMeans(n_clusters=k, max_iter=300, n_init=30, random_state=42)
        km.fit(X)
        inertia = float(km.inertia_)
        ssd.append(inertia)

        # Silhouette chỉ có ý nghĩa khi k >= 2
        if k >= 2:
            score = float(silhouette_score(X, km.labels_))
        else:
            score = float("nan")

        sil.append(score)
        rows.append({"k": k, "SSD": inertia, "Silhouette": score})

    metrics_df = pd.DataFrame(rows)

    # Lưu bảng chỉ số
    ssd_csv = out_dir / "kmeans_k_selection.csv"
    metrics_df.to_csv(ssd_csv, index=False, encoding="utf-8-sig")
    print(f"- K selection table saved to: {ssd_csv}")

    # Vẽ Elbow + Silhouette side-by-side (nếu muốn gọn, bạn có thể vẽ 2 hình riêng)
    fig, ax1 = plt.subplots(figsize=(9,6))
    ax1.plot(range_n_clusters, ssd, marker="o", label="SSD (inertia)")
    ax1.set_xlabel("Số cụm (k)")
    ax1.set_ylabel("SSD (inertia)")
    ax1.grid(True)

    # Trục phụ cho Silhouette
    ax2 = ax1.twinx()
    ax2.plot(range_n_clusters, sil, marker="s", linestyle="--", color="tab:orange", label="Silhouette")
    ax2.set_ylabel("Silhouette")

    # Gộp legend
    lines, labels = [], []
    for ax in (ax1, ax2):
        lns, lbs = ax.get_legend_handles_labels()
        lines += lns; labels += lbs
    ax1.legend(lines, labels, loc="best")

    kplot = out_dir / "kmeans_elbow_silhouette.png"
    plt.tight_layout()
    plt.savefig(kplot, dpi=150)
    plt.close()
    print(f"- Elbow + Silhouette plot saved to: {kplot}")

    # ---- Chọn k tốt nhất ----
    # Quy tắc đơn giản: ưu tiên k có Silhouette tối đa (k>=2).
    # Nếu có tie, lấy k nhỏ hơn để dễ diễn giải.
    valid = metrics_df[metrics_df["k"] >= 2].dropna(subset=["Silhouette"])
    if not valid.empty:
        k_best = int(valid.sort_values(["Silhouette","k"], ascending=[False, True]).iloc[0]["k"])
    else:
        # fallback nếu không tính được silhouette (hiếm)
        # lấy điểm “khuỷu” thô theo SSD: chọn k có giảm SSD tương đối lớn trước khi “chậm lại”.
        # Ở đây fallback đơn giản: chọn k=4 như trước.
        k_best = 4

    print(f"→ Chọn k tối ưu theo Silhouette: k={k_best}")

    # ---- Fit KMeans với k đã chọn và lưu kết quả ----
    km_best = KMeans(n_clusters=k_best, max_iter=500, n_init=50, random_state=42)
    rfm["cluster_k"] = km_best.fit_predict(X)

    clusters_csv = out_dir / f"kmeans_clusters_k{k_best}.csv"
    rfm[[cust_col, "cluster_k"] + feat_cols].to_csv(clusters_csv, index=False, encoding="utf-8-sig")
    print(f"- Cluster assignments (k={k_best}) saved to: {clusters_csv}")

    centers_unscaled = pd.DataFrame(
        scaler.inverse_transform(km_best.cluster_centers_),
        columns=feat_cols
    )
    centers_csv = out_dir / f"kmeans_centers_k{k_best}.csv"
    centers_unscaled.to_csv(centers_csv, index=False, encoding="utf-8-sig")
    print(f"- Cluster centers (unscaled) saved to: {centers_csv}")

    # In thêm Silhouette của k_best để tham khảo
    sil_best = float(silhouette_score(X, km_best.labels_)) if k_best >= 2 else float("nan")
    print(f"- Silhouette(k={k_best}) = {sil_best:.4f}")

    k = k_best

    kmeans = KMeans(n_clusters=k, n_init=30, max_iter=500, random_state=42)
    labels = kmeans.fit_predict(X)
    rfm["Cluster"] = labels

    # Silhouette (only if k>1)
    sil = silhouette_score(X, labels) if k > 1 else np.nan

    # Save assignments
    assign_cols = [cust_col, "Cluster"] + feat_cols
    assign_csv = out_dir / f"clusters_assignments_k{k_best}.csv"
    rfm[assign_cols].to_csv(assign_csv, index=False, encoding="utf-8-sig")

    # Save centers (unscaled)
    centers_unscaled = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=feat_cols
    )
    centers_unscaled["Cluster"] = range(len(centers_unscaled))
    centers_csv = out_dir / f"centers_unscaled_k{k_best}.csv"
    centers_unscaled.to_csv(centers_csv, index=False, encoding="utf-8-sig")

    # KPI by cluster
    kpi = rfm.groupby("Cluster").agg(
        customers=(cust_col, "nunique"),
        orders=("frequency", "sum"),
        total_sales=("monetary", "sum"),
        avg_order_value=("aov", "mean"),
        avg_frequency=("frequency","mean"),
        avg_recency_days=("recency_days","mean"),
        avg_discount=("avg_discount","mean") if "avg_discount" in rfm.columns else ("Cluster","size"),
        avg_ship_delay=("avg_ship_delay_days","mean") if "avg_ship_delay_days" in rfm.columns else ("Cluster","size"),
        total_profit=("total_profit","sum") if "total_profit" in rfm.columns else ("Cluster","size")
    ).reset_index()
    kpi_csv = out_dir / f"kpi_by_cluster_k{k}.csv"
    kpi.to_csv(kpi_csv, index=False, encoding="utf-8-sig")

    # Top products by cluster (if product column exists)
    top_prod_csv = None
    prod_col = find_col(df.columns, ["Product Name","Product","Product ID"])
    if prod_col and prod_col in df.columns:
        trans = df[[cust_col, prod_col, "Sales" if "Sales" in df.columns else sales_col]].copy()
        sales_used = "Sales" if "Sales" in trans.columns else sales_col
        trans = trans.merge(rfm[[cust_col, "Cluster"]], on=cust_col, how="left")
        tops = []
        for c in sorted(rfm["Cluster"].unique()):
            sub = trans[trans["Cluster"]==c]
            s = sub.groupby(prod_col)[sales_used].sum().sort_values(ascending=False).head(10)
            for prod, val in s.items():
                tops.append({"Cluster": int(c), "Product": str(prod), "TotalSales": float(val)})
        tp = pd.DataFrame(tops)
        top_prod_csv = out_dir / f"top_products_by_cluster_k{k}.csv"
        tp.to_csv(top_prod_csv, index=False, encoding="utf-8-sig")

    # Visualizations
    # Pairplot (if seaborn)
    if sns is not None:
        try:
            sns.set(style="ticks")
            pp = sns.pairplot(rfm, hue="Cluster", vars=feat_cols, palette="Set2", corner=True)
            pp.fig.suptitle("Quan hệ cụm – RFM (recency_days, frequency, monetary, aov)", y=1.02)
            pairplot_png = out_dir / f"pairplot_k{k}.png"
            pp.savefig(pairplot_png, dpi=150)
            plt.close('all')
        except Exception as e:
            print(f"[WARN] Không thể vẽ pairplot: {e}")

    # KPI bar plot
    try:
        plt.figure(figsize=(10,6))
        metrics = ["total_sales","avg_order_value","avg_frequency","avg_recency_days"]
        mlong = kpi.melt(id_vars="Cluster", value_vars=metrics, var_name="metric", value_name="value")
        if sns is not None:
            sns.barplot(data=mlong, x="metric", y="value", hue="Cluster")
        else:
            # fallback: simple multi-bar
            clusters = sorted(kpi["Cluster"].unique())
            x = np.arange(len(metrics))
            width = 0.8/len(clusters)
            for i, c in enumerate(clusters):
                vals = [kpi.loc[kpi["Cluster"]==c, m].values[0] for m in metrics]
                plt.bar(x + i*width, vals, width=width, label=f"Cluster {c}")
            plt.xticks(x + width*(len(clusters)-1)/2, metrics)
            plt.legend()
        plt.title("So sánh KPI theo cụm")
        plt.xticks(rotation=15)
        plt.tight_layout()
        kpi_bar_png = out_dir / f"kpi_bar_k{k}.png"
        plt.savefig(kpi_bar_png, dpi=150)
        plt.close('all')
    except Exception as e:
        print(f"[WARN] Không thể vẽ KPI bar: {e}")

    print(f"[OK] Saved: {assign_csv}")
    print(f"[OK] Saved: {centers_csv}")
    print(f"[OK] Saved: {kpi_csv}")
    if top_prod_csv: print(f"[OK] Saved: {top_prod_csv}")
    if sns is not None: print(f"[OK] Pairplot saved (nếu seaborn có sẵn)")
    print(f"[OK] Silhouette score (k={k}): {sil:.4f}" if k>1 else "[OK] k=1, không tính silhouette")

if __name__ == "__main__":
    main()
