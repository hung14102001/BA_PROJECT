from datetime import timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
import sys
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def find_col(cols, candidates):
    for cand in candidates:
        for c in cols:
            if str(c).strip().lower() == cand.strip().lower():
                return c
    return None

def read_csv_fallback(p: Path):
    encs = ["utf-8", "cp1252", "latin1"]
    for e in encs:
        try:
            return pd.read_csv(p, encoding=e, low_memory=False)
        except Exception:
            continue
    return pd.read_csv(p, low_memory=False)

def safe_save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)

def main():
    # project = Path(__file__).resolve().parents[2]
    if sns is not None:
        sns.set_theme(style="whitegrid")
    inp = Path("./data/sample_age.csv")
    if not inp.exists():
        print(f"Không tìm thấy file: {inp}", file=sys.stderr)
        sys.exit(1)

    df = read_csv_fallback(inp)
    df.columns = [str(c).strip() for c in df.columns]

    sales_col = find_col(df.columns, ["Sales", "sales", "Revenue", "Amount"])
    profit_col = find_col(df.columns, ["Profit", "profit"])
    qty_col = find_col(df.columns, ["Quantity", "Qty", "Order Quantity", "Order Qty"])
    cust_col = find_col(df.columns, ["Customer ID", "CustomerID", "Customer"])
    date_col = find_col(df.columns, ["Order Date", "OrderDate", "Date", "Ship Date"])
    market_col = find_col(df.columns, ["Country", "Market", "State", "Region"])
    segment_col = find_col(df.columns, ["Segment", "segment"])
    product_col = find_col(df.columns, ["Product Name", "Product", "Product ID", "ProductID"])
    membership_col = find_col(df.columns, ["membership", "Membership", "Tier"])
    state_col = find_col(df.columns, ["State", "Province", "State/Province", "Region", "Territory"])

    if sales_col is None:
        print("Không tìm thấy cột Sales.", file=sys.stderr)
        sys.exit(1)

    # normalize
    # Chuẩn hóa dữ liệu số, loai bỏ ký tự $ và dấu phẩy
    df[sales_col] = df[sales_col].astype(str).str.replace(r'[\$,]', '', regex=True)
    df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce').fillna(0.0)
    if profit_col:
        df[profit_col] = pd.to_numeric(df[profit_col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0.0)
    if qty_col:
        df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)

    # dates
    # xử lý cột ngày tháng
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        df["_date_"] = pd.to_datetime("1970-01-01")
        date_col = "_date_"
    df["year_month"] = df[date_col].dt.to_period("M").astype(str)

    out_dir =  Path("./analysis_outputs")
    out_dir.mkdir(exist_ok=True)

    # • What is the total number of sales?
    # • What is the average sales per month?
    # • What is the monthly revenue?
    # • What are the key demographics of the customers?
    # • Which market (country) generated the most sales on average?
    # • What were the profits by segment?
    # • When were the best- and worst-selling periods?
    # • Which products sell best?
    # • Which products should the company order more or less of?
    # • How should the company adjust its marketing strategies to VIP customers and less-engaged ones?
    # • Should the company acquire new customers, and how much money should they spend on it?

    # 1) Doanh số & số giao dịch theo năm (2 trục Oy)
    df["year"] = df[date_col].dt.year.astype(int)  # ép thành số nguyên

    yearly_rev = df.groupby("year")[sales_col].sum().sort_index()
    yearly_trans = df.groupby("year")[sales_col].count().sort_index()

    fig, ax1 = plt.subplots(figsize=(10,6))

    # Vẽ cột doanh số (trục trái)
    bars = ax1.bar(yearly_rev.index, yearly_rev.values, color="lightgreen", width=0.4, label="Doanh số")
    ax1.set_ylabel("Doanh số", fontsize=12)
    ax1.set_xlabel("Năm")
    ax1.set_title("Doanh số & Số giao dịch theo năm", fontsize=16, weight="bold")

    # Ghi nhãn doanh số trên cột
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height,
                 f"{int(height):,}", ha="center", va="bottom", fontsize=9)

    # Vẽ số giao dịch (line chart, trục phải)
    ax2 = ax1.twinx()
    ax2.plot(yearly_trans.index, yearly_trans.values,
             color="blue", marker="o", linewidth=2, label="Số giao dịch")
    ax2.set_ylabel("Số giao dịch", fontsize=12, color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    # Ghi nhãn số giao dịch trên từng điểm
    for x, y in zip(yearly_trans.index, yearly_trans.values):
        ax2.text(x, y, f"{y:,}", color="blue", fontsize=9, ha="center", va="bottom")

    # Box tổng ở góc trên trái
    total_sales = yearly_rev.sum()
    total_trans = yearly_trans.sum()
    ax1.text(
        0.01, 0.95,
        f"Tổng giao dịch: {total_trans:,}\nTổng doanh số: {total_sales:,.2f}",
        ha="left", va="center",
        transform=ax1.transAxes,
        fontsize=12, color="black",
        bbox=dict(facecolor="lightyellow", edgecolor="gray", boxstyle="round,pad=0.5")
    )

    # Làm tròn trục X thành năm nguyên
    ax1.set_xticks(yearly_rev.index)
    ax1.set_xticklabels(yearly_rev.index.astype(int))

    fig_path5 = out_dir / "Cau1.png"
    plt.tight_layout()
    plt.savefig(fig_path5, dpi=150)
    print(f"- Biểu đồ doanh số & số giao dịch (2 trục Oy) lưu tại: {fig_path5}")

    # 2) Doanh thu hàng tháng và doanh số trung bình mỗi tháng là bao nhiêu?

    monthly_rev = df.groupby("year_month")[sales_col].sum().sort_index()

    total_sales = float(df[sales_col].sum())
    avg_sales_per_month =  float(monthly_rev.mean()) if len(monthly_rev)>0 else 0.0

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x=monthly_rev.index, y=monthly_rev.values, color="skyblue")

    # Vẽ đường trung bình
    plt.axhline(avg_sales_per_month, color="red", linestyle="--", label=f"Mean: {avg_sales_per_month:,.2f}")
    
    # Gắn nhãn
    plt.xticks(rotation=45, ha="right")
    plt.title("Doanh số theo tháng", fontsize=16, weight="bold")
    plt.xlabel("Tháng")
    plt.ylabel("Doanh số")
    plt.legend()

    # Lưu biểu đồ
    fig_path = out_dir / "Cau2_3.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    print(f"- Biểu đồ doanh số theo tháng được lưu tại: {fig_path}")

    # # 2) Sales distribution (per transaction)
    # fig, ax = plt.subplots(figsize=(8,4))
    # if sns:
    #     sns.histplot(df[sales_col], bins=80, kde=False, ax=ax)
    # else:
    #     ax.hist(df[sales_col].dropna(), bins=80)
    # ax.set_title("Distribution of Sales per Transaction")
    # ax.set_xlabel("Sales")
    # ax.set_ylabel("Count")
    # safe_save_fig(fig, out_dir / "sales_distribution.png")

    # 3) Đặc điểm nhân khẩu học chính của khách hàng là gì?

    if "age" in df.columns:
        sns.set_theme(style="whitegrid")
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

        age_bins = [18, 25, 35, 45, 55, 65, 80]
        age_lbls = [f"{age_bins[i]}–{age_bins[i+1]-1}" for i in range(len(age_bins)-1)]
        df["age_group"] = pd.cut(df["age"], bins=age_bins)

        agg = df.groupby("age_group").agg(
            total_sales=(sales_col, "sum"),     # Tổng doanh số theo nhóm tuổi
            transactions=("age", "size")        # Số giao dịch theo nhóm tuổi
        ).reindex(pd.IntervalIndex.from_breaks(age_bins), fill_value=0)

        fig, ax1 = plt.subplots(figsize=(12, 5))

        # Bar: Tổng doanh số (trục trái)
        bars = ax1.bar(range(len(agg)), agg["total_sales"].values, color="#8fd19e",
                    width=0.6, label="Tổng doanh số")
        ax1.set_ylabel("Tổng doanh số")
        ax1.set_xlabel("Nhóm tuổi")
        ax1.set_xticks(range(len(agg)))
        ax1.set_xticklabels(age_lbls)

        # Nhãn trên cột (doanh số)
        for b, v in zip(bars, agg["total_sales"].values):
            ax1.text(b.get_x() + b.get_width()/2, v, f"{v:,.0f}",
                    ha="center", va="bottom", fontsize=9)

        # Line: Số giao dịch (trục phải)
        ax2 = ax1.twinx()
        ax2.plot(range(len(agg)), agg["transactions"].values, color="#1f77b4",
                marker="o", linewidth=2.2, label="Số giao dịch")
        ax2.set_ylabel("Số giao dịch", color="#1f77b4")
        ax2.tick_params(axis="y", labelcolor="#1f77b4")

        # THÊM NHÃN SỐ GIAO DỊCH TRÊN TỪNG ĐIỂM
        for x, y in enumerate(agg["transactions"].values):
            ax2.annotate(f"{int(y):,}", (x, y), textcoords="offset points",
                        xytext=(0, 6), ha="center", color="#1f77b4", fontsize=9)

        # TẮT ĐƯỜNG NGANG (GRID) CỦA TRỤC PHẢI
        ax2.grid(False)                # tắt toàn bộ grid của ax2

        plt.title("Tổng doanh số & Số giao dịch theo nhóm tuổi", fontsize=16, weight="bold")

        # Legend gộp 2 trục
        handles, labels = [], []
        for a in (ax1, ax2):
            h, l = a.get_legend_handles_labels()
            handles += h; labels += l
        ax1.legend(handles, labels, loc="upper right")

        fig_path = out_dir / "cau4.png"
        plt.tight_layout()
        try:
            safe_save_fig(fig, fig_path)
        except Exception:
            plt.savefig(fig_path, dpi=150)
        print(f"- Biểu đồ tuổi (Doanh số & Số giao dịch) lưu tại: {fig_path}")
    # 7) Thị trường (quốc gia) nào tạo ra doanh số trung bình cao nhất?

    state_col = find_col(df.columns, ["State", "Province", "State/Province", "State Province"])
    if state_col is not None:
        # Tổng hợp theo state
        state_stats = df.groupby(state_col)[sales_col].agg(
            total_sale="sum",
            avg_sale="mean",
            transactions="count"
        )

        # Lấy Top 10 theo tổng doanh số
        top_states = state_stats.sort_values("total_sale", ascending=False).head(10)

        # Chuẩn bị toạ độ trục X rời rạc để vẽ bar + line chung
        x_labels = top_states.index.astype(str)
        x = np.arange(len(top_states))

        # Vẽ
        fig, ax1 = plt.subplots(figsize=(12, 5))

        # Bar: Tổng doanh số (trục trái)
        if 'sns' in globals() and sns is not None:
            sns.set_theme(style="whitegrid")
        ax1.bar(x, top_states["total_sale"].values, color="#8fd19e", width=0.6, label="Total sales")
        ax1.set_title("Top 10 State/Province by TOTAL Sales", fontsize=14, weight="bold")
        ax1.set_xlabel("State / Province")
        ax1.set_ylabel("Total Sales")
        ax1.set_xticks(x)
        ax1.set_xticklabels(x_labels, rotation=30, ha="right")

        # Nhãn trên cột: total
        for xi, total_v in zip(x, top_states["total_sale"].values):
            ax1.text(xi, total_v, f"{total_v:,.0f}", ha="center", va="bottom", fontsize=10)

        # Line: Transactions (trục phải)
        ax2 = ax1.twinx()
        ax2.plot(x, top_states["transactions"].values, color="#1f77b4",
                marker="o", linewidth=2, label="Transactions")
        ax2.set_ylabel("Transactions", color="#1f77b4")
        ax2.tick_params(axis="y", labelcolor="#1f77b4")
        ax2.grid(False)  # bỏ grid của trục phải để chart sạch hơn

        # Thêm nhãn số giao dịch trên từng điểm
        for xi, y in zip(x, top_states["transactions"].values):
            ax2.annotate(f"{int(y):,}", (xi, y), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=9, color="#1f77b4")

        # Legend gộp 2 trục
        handles, labels = [], []
        for a in (ax1, ax2):
            h, l = a.get_legend_handles_labels()
            handles += h; labels += l
        ax1.legend(handles, labels, loc="upper right")

        # Lưu hình
        fig_path = out_dir / "cau5.png"
        plt.tight_layout()
        try:
            safe_save_fig(fig, fig_path)
        except Exception:
            plt.savefig(fig_path, dpi=150)
        print(f"- Top 10 state/province by TOTAL sales + transactions saved to: {fig_path}")

        # In state đứng đầu theo tổng doanh số
        best_state = top_states["total_sale"].idxmax()
        best_total = top_states.loc[best_state, "total_sale"]
        best_avg   = top_states.loc[best_state, "avg_sale"]
        best_n     = int(top_states.loc[best_state, "transactions"])
        print(f"- State with highest TOTAL sales: {best_state} "
            f"(total={best_total:,.0f}, avg/order={best_avg:,.2f}, transactions={best_n:,})")
    else:
        print("- Không tìm thấy cột State/Province để tính tổng doanh số theo state.")


    # 6)  Lợi nhuận theo từng phân khúc là bao nhiêu?

    if profit_col and segment_col:
        seg = df.groupby(segment_col)[profit_col].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(seg.index.astype(str), seg.values)
        ax.set_title("Profits by Segment")
        ax.set_ylabel("Profit")
        ax.tick_params(axis="x", rotation=45)
        safe_save_fig(fig, out_dir / "Cau6.png")

    # 7) Giai đoạn nào bán chạy nhất và kém nhất?
    mm = df.groupby(["year_month"])[sales_col].sum()
    if not mm.empty:
        # convert year_month to pivot table year x month
        ym = pd.to_datetime(mm.index.to_series().astype(str) + "-01", errors="coerce")
        tmp = pd.DataFrame({"ym": mm.index, "date": ym, "revenue": mm.values})
        tmp["year"] = tmp["date"].dt.year
        tmp["month"] = tmp["date"].dt.month
        pivot = tmp.pivot_table(index="year", columns="month", values="revenue", aggfunc="sum").fillna(0)
        fig, ax = plt.subplots(figsize=(10,4))
        if sns:
            sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
        else:
            im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu")
            fig.colorbar(im, ax=ax)
            ax.set_xticks(np.arange(pivot.shape[1]))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(np.arange(pivot.shape[0]))
            ax.set_yticklabels(pivot.index)
        ax.set_title("Monthly Revenue Heatmap (year x month)")
        safe_save_fig(fig, out_dir / "Cau7.png")

    # 8) Sản phẩm nào bán chạy nhất?
    if product_col and qty_col:
        prod_agg = df.groupby(product_col).agg(
            total_units=(qty_col, "sum"),      # số lượng bán ra
            total_sales=(sales_col, "sum")     # tổng doanh số của sản phẩm đó
        )

        top10_qty = prod_agg.sort_values("total_units", ascending=False).head(10)

        # Vẽ bar ngang: trục = total_units; nhãn bên phải = total_sales
        fig, ax = plt.subplots(figsize=(11, 6))
        names = top10_qty.index.astype(str)
        units = top10_qty["total_units"].values
        sales = top10_qty["total_sales"].values

        ax.barh(names, units, color="#8fd19e")
        ax.invert_yaxis()  # sản phẩm bán chạy nhất lên trên
        ax.set_title("Top 10 Products by Quantity Sold (with Total Sales)", fontsize=14, weight="bold")
        ax.set_xlabel("Units Sold")

        # gắn nhãn doanh số ở cuối mỗi thanh
        for y, u, s in zip(range(len(names)), units, sales):
            ax.text(u, y, f"  Total Sales: {s:,.0f}", ha="left", va="center", fontsize=9, color="#1f77b4")

        plt.tight_layout()
        safe_save_fig(fig, out_dir / "top10_products_by_quantity_with_sales.png")

        print(f"- Chart: {out_dir / 'top10_products_by_quantity_with_sales.png'}")
    elif product_col and not qty_col:
        print("⚠️ Không có cột Quantity/Qty → không thể xếp hạng theo số lượng. Hãy bổ sung cột số lượng.")
    else:
        print("⚠️ Thiếu cột Product để tính Top 10 theo số lượng.")

    # 9) Công ty nên đặt hàng nhiều hơn hay ít hơn những sản phẩm nào?

    if product_col:
        has_qty = qty_col is not None

        # Chuẩn bị dữ liệu theo tháng
        df["_ym"] = df[date_col].dt.to_period("M") if date_col else pd.PeriodIndex([], freq="M")
        agg = df.groupby(product_col).agg(
            total_units=(qty_col, "sum") if has_qty else (sales_col, "count"),
            total_sales=(sales_col, "sum"),
            transactions=(sales_col, "count"),
            months_active=("_ym", lambda s: s.nunique())
        )

        # Velocity theo tháng
        agg["units_per_month"] = agg["total_units"] / agg["months_active"].replace(0, np.nan)
        agg["sales_per_month"] = agg["total_sales"] / agg["months_active"].replace(0, np.nan)
        agg = agg.fillna(0)

        # Lọc độ tin cậy tối thiểu (nếu rỗng sau lọc thì dùng toàn bộ)
        base = agg[(agg["months_active"] >= 3) & (agg["transactions"] >= 5)]
        if base.empty:
            base = agg.copy()

        # Chọn metric: ưu tiên units_per_month nếu có Quantity
        metric = "units_per_month" if has_qty else "sales_per_month"

        # Ngưỡng phân vị 20% / 80%
        p20 = np.percentile(base[metric], 20) if len(base) else 0
        p80 = np.percentile(base[metric], 80) if len(base) else 0

        # Gán action
        base["action"] = np.where(
            base[metric] >= p80, "ORDER_MORE",
            np.where(base[metric] <= p20, "ORDER_LESS", "HOLD")
        )

        # Lấy top 10 mỗi nhóm để báo cáo nhanh
        more = base.query("action == 'ORDER_MORE'").sort_values(metric, ascending=False).head(10)
        less = base.query("action == 'ORDER_LESS'").sort_values(metric, ascending=True).head(10)

        # Lưu bảng khuyến nghị
        out_tbl = pd.concat([more.assign(priority="HIGH"), less.assign(priority="LOW")], axis=0)
        print("- ORDER MORE (top 10):", list(more.index))
        print("- ORDER LESS  (top 10):", list(less.index))

        # Biểu đồ minh hoạ 2 panel
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        def plot_panel(ax, data, title, color):
            if data.empty:
                ax.axis("off"); ax.set_title(title + " (none)"); return
            names = data.index.astype(str)
            vals = data[metric].values
            ax.barh(names, vals, color=color)
            ax.invert_yaxis()
            ax.set_title(title)
            ax.set_xlabel(metric.replace("_", " ").title())
            for y, v in enumerate(vals):
                ax.text(v, y, f"  {v:,.2f}" if metric.endswith("per_month") else f"  {int(v):,}", va="center", ha="left", fontsize=9)

        plot_panel(axes[0], more, "ORDER MORE (Top 10)", "#8fd19e")
        plot_panel(axes[1], less, "ORDER LESS (Top 10)", "#f4a261")
        plt.tight_layout()
        safe_save_fig(fig, out_dir / "cau9.png")



    # Kiểm tra cột cần thiết
    # • Công ty nên điều chỉnh chiến lược tiếp thị của mình như thế nào đối với khách hàng VIP và khách hàng ít tương tác?
    # • Công ty có nên thu hút khách hàng mới không và nên chi bao nhiêu tiền cho việc này?

    # Chuẩn hoá dữ liệu nguồn
    df = df.copy()

    # Tạo đặc trưng RFM per customer
    today = df[date_col].max() + timedelta(days=1)
    rfm = df.groupby(cust_col).agg(
        last_date=(date_col, "max"),
        frequency=(cust_col, "size"),
        monetary=(sales_col, "sum"),
        aov=(sales_col, "mean")
    ).reset_index()
    rfm["recency_days"] = (today - rfm["last_date"]).dt.days

    # Ma trận đặc trưng để cluster
    feat_cols = ["recency_days","frequency","monetary","aov"]
    X_raw = rfm[feat_cols].fillna(0.0)

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)


    # ---- Tìm k tối ưu: so sánh SSD (Elbow) + Silhouette ----
    from sklearn.metrics import silhouette_score

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


    # # Elbow: SSD (inertia_) cho k = 1..8
    # range_n_clusters = [1,2,3,4,5,6,7,8]
    # ssd = []
    # for k in range_n_clusters:
    #     km = KMeans(n_clusters=k, max_iter=200, n_init=20, random_state=42)
    #     km.fit(X)
    #     ssd.append(float(km.inertia_))

    # # Vẽ Elbow curve
    # fig, ax = plt.subplots(figsize=(8,6))
    # ax.plot(range_n_clusters, ssd, marker="o")
    # ax.set_title("Elbow Curve for K-Means Clustering")
    # ax.set_xlabel("Số cụm (k)")
    # ax.set_ylabel("Sum of Squared Distances (SSD)")
    # ax.grid(True)

    # elbow_png = out_dir / "kmeans_elbow.png"
    # plt.tight_layout()
    # try:
    #     safe_save_fig(fig, elbow_png)
    # except Exception:
    #     plt.savefig(elbow_png, dpi=150)
    # print(f"- Elbow curve saved to: {elbow_png}")

    # # # Lưu bảng SSD để tham khảo
    # # ssd_df = pd.DataFrame({"k": range_n_clusters, "SSD": ssd})
    # # ssd_csv = out_dir / "kmeans_elbow_ssd.csv"
    # # ssd_df.to_csv(ssd_csv, index=False, encoding="utf-8-sig")
    # # print(f"- SSD table saved to: {ssd_csv}")

    # # (Tuỳ chọn) chọn k và gán nhãn cluster để bạn xem thử
    # # Ở đây mình ví dụ k=4; bạn nhìn Elbow rồi đổi con số này tuỳ theo “điểm khuỷu”
    # k_pick = 4
    # km4 = KMeans(n_clusters=k_pick, max_iter=300, n_init=30, random_state=42)
    # rfm["cluster_k4"] = km4.fit_predict(X)

    # # Lưu cluster assignment + tâm cụm để kiểm tra
    # clusters_csv = out_dir / f"kmeans_clusters_k{k_pick}.csv"
    # rfm[[cust_col, "cluster_k4"] + feat_cols].to_csv(clusters_csv, index=False, encoding="utf-8-sig")
    # print(f"- Cluster assignments (k={k_pick}) saved to: {clusters_csv}")

    # centers = pd.DataFrame(km4.cluster_centers_, columns=feat_cols)
    # # đảo scale về đơn vị gốc để dễ hiểu
    # centers_unscaled = pd.DataFrame(scaler.inverse_transform(km4.cluster_centers_), columns=feat_cols)
    # centers_csv = out_dir / f"kmeans_centers_k{k_pick}.csv"
    # centers_unscaled.to_csv(centers_csv, index=False, encoding="utf-8-sig")
    # print(f"- Cluster centers (unscaled) saved to: {centers_csv}")




    # # print("Đã sinh biểu đồ vào:", out_dir)

if __name__ == "__main__":
    main()