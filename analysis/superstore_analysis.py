import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import dataframe_image as dfi


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
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)

def main():
    if sns is not None:
        sns.set_theme(style="whitegrid")
    inp = Path("data/dsSuperstore.csv")
    if not inp.exists():
        sys.exit(1)

    df = read_csv_fallback(inp)
    df.columns = [str(c).strip() for c in df.columns]

    sales_col = find_col(df.columns, ["Sales", "sales", "Revenue", "Amount"])
    profit_col = find_col(df.columns, ["Profit", "profit"])
    qty_col = find_col(df.columns, ["Quantity", "Qty", "Order Quantity", "Order Qty"])
    cust_col = find_col(df.columns, ["Customer ID", "CustomerID", "Customer"])
    date_col = find_col(df.columns, ["Order Date", "OrderDate", "Date", "Ship Date"])
    segment_col = find_col(df.columns, ["Segment", "segment"])
    product_col = find_col(df.columns, ["Product Name", "Product", "Product ID", "ProductID"])
    state_col = find_col(df.columns, ["State", "Province"])
    disc_col = find_col(df.columns, ["Discount", "discount"])

    if sales_col is None:
        print("Không tìm thấy cột Sales.", file=sys.stderr)
        sys.exit(1)

    # normalize
    df[sales_col] = df[sales_col].astype(str).str.replace(r'[\$,]', '', regex=True)
    df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce').fillna(0.0)
    if profit_col:
        df[profit_col] = pd.to_numeric(df[profit_col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0.0)
    if qty_col:
        df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
    if df[disc_col].max() <= 1.5:
        df[disc_col] = (df[disc_col] * 100).round(1)

    # dates
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        df["_date_"] = pd.to_datetime("1970-01-01")
        date_col = "_date_"
    df["year_month"] = df[date_col].dt.to_period("M").astype(str)

    out_dir =  Path("./analysis_outputs")
    out_dir.mkdir(exist_ok=True)

    df.to_csv(Path("data/cleaned_data.csv"), index=False)

    # Box plot + Histogram cho các cột số chính
    cols = ["Sales", "Quantity", "Discount", "Profit"]

    fig, axes = plt.subplots(2, len(cols), figsize=(4 * len(cols), 8))

    for i, col in enumerate(cols):
        # Boxplot
        sns.boxplot(y=df[col], ax=axes[1, i])
        axes[1, i].set_title(f"Boxplot cột {col}")
        
        # Histogram
        sns.histplot(df[col], bins=30, kde=True, ax=axes[0, i])
        axes[0, i].set_title(f"Histogram cột {col}")
        

    plt.tight_layout()
    # plt.show()
    safe_save_fig(fig, out_dir / "Box_plot_histogram.png")
    # plt.savefig(fig_path, dpi=200)

    # Bảng thống kê mô tả
    summary = df.describe().T
    styled = (summary.style
            .set_table_styles([
                {'selector': 'th', 'props': [('font-size', '14pt'), ('text-align', 'center')]},
                {'selector': 'td', 'props': [('font-size', '13pt'), ('text-align', 'center')]},
                {'selector': 'table', 'props': [('border', '1px solid black'), ('border-collapse', 'collapse')]}
            ])
            .set_caption("Bảng thống kê mô tả")
            )
    # Xuất ra ảnh PNG
    dfi.export(styled, out_dir / "Summary_table.png", dpi=200)


    # 1) Tổng và doanh số - số giao dịch theo năm
    df["year"] = df[date_col].dt.year.astype(int)  

    yearly_rev = df.groupby("year")[sales_col].sum().sort_index()
    yearly_trans = df.groupby("year")[sales_col].count().sort_index()

    fig, ax1 = plt.subplots(figsize=(10,6))

    bars = ax1.bar(yearly_rev.index, yearly_rev.values, color="lightgreen", width=0.4, label="Doanh số")
    ax1.set_ylabel("Doanh số ($)", fontsize=12)
    ax1.set_xlabel("Năm")
    ax1.set_title("Doanh số & Số giao dịch theo năm", fontsize=16, weight="bold")

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height,
                 f"{int(height):,}", ha="center", va="bottom", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(yearly_trans.index, yearly_trans.values,
             color="blue", marker="o", linewidth=2, label="Số giao dịch")
    ax2.set_ylabel("Số giao dịch", fontsize=12, color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    for x, y in zip(yearly_trans.index, yearly_trans.values):
        ax2.text(x, y, f"{y:,}", color="blue", fontsize=9, ha="center", va="bottom")

    total_sales = yearly_rev.sum()
    total_trans = yearly_trans.sum()
    ax1.text(
        0.01, 0.95,
        f"Tổng giao dịch: {total_trans:,}\nTổng doanh số: {int(total_sales):,} $",
        ha="left", va="center",
        transform=ax1.transAxes,
        fontsize=12, color="black",
        bbox=dict(facecolor="lightyellow", edgecolor="gray", boxstyle="round,pad=0.5")
    )

    ax1.set_xticks(yearly_rev.index)
    ax1.set_xticklabels(yearly_rev.index.astype(int))

    fig_path = out_dir / "Sum_Revenue.png"
    plt.tight_layout()
    safe_save_fig(plt.gcf(), fig_path)

    # 2) Doanh thu hàng tháng và doanh số trung bình mỗi tháng là bao nhiêu?

    monthly_rev = df.groupby("year_month")[sales_col].sum().sort_index()

    total_sales = float(df[sales_col].sum())
    avg_sales_per_month =  float(monthly_rev.mean()) if len(monthly_rev)>0 else 0.0

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x=monthly_rev.index, y=monthly_rev.values, color="skyblue")

    # Trung bình tháng
    plt.axhline(avg_sales_per_month, color="red", linestyle="--", label=f"Trung bình tháng: {int(avg_sales_per_month):,} $")
    
    plt.xticks(rotation=45, ha="right")
    plt.title("Doanh số theo tháng ($)", fontsize=16, weight="bold")
    plt.xlabel("Tháng")
    plt.ylabel("Doanh số")
    plt.legend()

    # Lưu biểu đồ
    fig_path = out_dir / "Monthly_revenue.png"
    plt.tight_layout()
    safe_save_fig(plt.gcf(), fig_path)

    
    # 3) Phân tích khách hàng theo phân khúc (Segment): doanh số & lợi nhuận
    grouped = df.groupby("Segment")[["Sales", "Profit"]].sum().sort_values("Sales", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))

    bar_width = 0.35
    x = range(len(grouped))

    # Cột Sales
    ax.bar([i - bar_width/2 for i in x], grouped["Sales"], 
        width=bar_width, label="Doanh số", color="#6EC1E4")

    # Cột Profit
    ax.bar([i + bar_width/2 for i in x], grouped["Profit"], 
        width=bar_width, label="Lợi nhuận", color="#F78CA0")

    # Thêm số giá trị trên mỗi cột
    for i, (s, p) in enumerate(zip(grouped["Sales"], grouped["Profit"])):
        ax.text(i - bar_width/2, s + 20000, f"{int(s):,}", ha='center', va='bottom', fontsize=9, color='black')
        ax.text(i + bar_width/2, p + 20000, f"{int(p):,}", ha='center', va='bottom', fontsize=9, color='black')

    # Cấu hình hiển thị
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index, rotation=20, fontsize=11)
    ax.set_ylabel("Giá trị ($)", fontsize=11)
    ax.set_title("Doanh số và lợi nhuận theo từng phân khúc khách hàng", fontsize=13, fontweight='bold')
    ax.legend()

    # Format trục Y để không hiển thị 1e6
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    # plt.show()
    safe_save_fig(fig, out_dir / "Segment_customer.png")

    # 4) Thị trường (bang nào) tạo ra doanh số trung bình cao nhất?
    # Tổng hợp theo state
    state_stats = df.groupby(state_col)[sales_col].agg(
        total_sale="sum",
        avg_sale="mean",
        transactions="count"
    )

    # Lấy Top 10 theo tổng doanh số
    top_states = state_stats.sort_values("total_sale", ascending=False).head(10)

    x_labels = top_states.index.astype(str)
    x = np.arange(len(top_states))

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Bar: Tổng doanh số (trục trái)
    if 'sns' in globals() and sns is not None:
        sns.set_theme(style="whitegrid")
    ax1.bar(x, top_states["total_sale"].values, color="#8fd19e", width=0.6, label="Tổng doanh số")
    ax1.set_title("Top 10 thị trường theo doanh số", fontsize=14, weight="bold")
    ax1.set_xlabel("Bang")
    ax1.set_ylabel("Tổng doanh số ($)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=30, ha="right")

    # Nhãn trên cột: total
    for xi, total_v in zip(x, top_states["total_sale"].values):
        ax1.text(xi, total_v, f"{total_v:,.0f}", ha="center", va="bottom", fontsize=10)

    # Line: Transactions (trục phải)
    ax2 = ax1.twinx()
    ax2.plot(x, top_states["transactions"].values, color="#1f77b4",
            marker="o", linewidth=2, label="Số giao dịch")
    ax2.set_ylabel("Số giao dịch", color="#1f77b4")
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
    plt.tight_layout()
    safe_save_fig(fig, out_dir / "Top10State.png")

    # 5) Phân tích nhóm mặt hàng (Sub-Category): doanh số & lợi nhuận
    subcat_group = df.groupby("Sub-Category")[["Sales", "Profit"]].sum().sort_values("Sales", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))

    bar_width = 0.35
    x = range(len(subcat_group))

    # Sales
    ax.bar([i - bar_width/2 for i in x], subcat_group["Sales"], 
        width=bar_width, label="Sales", color="#6EC1E4")

    # Profit
    ax.bar([i + bar_width/2 for i in x], subcat_group["Profit"], 
        width=bar_width, label="Profit", color="#F78CA0")

    for i, (s, p) in enumerate(zip(subcat_group["Sales"], subcat_group["Profit"])):
        # Hiển thị giá trị Sales
        ax.text(i - bar_width/2, s + s*0.01, f"{int(s):,}", 
                ha='center', va='bottom', fontsize=8, color='black')
        
        # Hiển thị giá trị Profit (tự căn vị trí nếu âm)
        ax.text(i + bar_width/2, p + abs(p)*0.02 if p > 0 else p - abs(p)*0.15, 
                f"{int(p):,}", ha='center', va='bottom', fontsize=8, color='black')


    # Thiết lập trục và tiêu đề
    ax.set_xticks(x)
    ax.set_xticklabels(subcat_group.index, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel("Giá trị ($)", fontsize=11)
    ax.set_title("Doanh số và lợi nhuận theo từng nhóm mặt hàng", fontsize=13, fontweight='bold')
    ax.legend()

    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    # plt.show()
    safe_save_fig(fig, out_dir / "Sales_Profit_by_SubCategory.png")

    # 6) Giai đoạn nào bán chạy nhất và kém nhất?
    mm = df.groupby(["year_month"])[sales_col].sum()
    if not mm.empty:
        # convert year_month to pivot table year x month
        ym = pd.to_datetime(mm.index.to_series().astype(str) + "-01", errors="coerce")
        tmp = pd.DataFrame({"ym": mm.index, "date": ym, "revenue": mm.values})
        tmp["year"] = tmp["date"].dt.year
        tmp["month"] = tmp["date"].dt.month
        pivot = tmp.pivot_table(index="year", columns="month", values="revenue", aggfunc="sum").fillna(0)
        fig, ax = plt.subplots(figsize=(10,4))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
        ax.set_title("Doanh số theo tháng (năm x tháng) - Heatmap")
        ax.set_xlabel("Tháng", fontsize=11)
        ax.set_ylabel("Năm", fontsize=11)
        safe_save_fig(fig, out_dir / "Heatmap_revenue.png")

    # 7) Sản phẩm nào bán chạy nhất?
    grouped = (
        df.groupby("Product ID")[["Quantity", "Sales", "Profit"]]
        .sum()
        .sort_values(by="Sales", ascending=False)
    )

    # Lấy top 5 sản phẩm bán chạy nhất
    top5 = grouped.head(5).reset_index()

    # --- Vẽ biểu đồ ---
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Biểu đồ cột cho Sales và Profit
    bar_width = 0.35
    x = range(len(top5))
    ax1.bar([i - bar_width/2 for i in x], top5["Sales"], width=bar_width, label="Doanh số", color="skyblue")
    ax1.bar([i + bar_width/2 for i in x], top5["Profit"], width=bar_width, label="Lợi nhuận", color="salmon")

    # Thêm nhãn giá trị trên cột
    for i, (s, p) in enumerate(zip(top5["Sales"], top5["Profit"])):
        ax1.text(i - bar_width/2, s + s*0.02, f"{s:,.0f}", ha='center', va='bottom', fontsize=9, color='black')
        ax1.text(i + bar_width/2, p + (p*0.02 if p >= 0 else p - abs(p)*0.15), 
                f"{p:,.0f}", ha='center', va='bottom', fontsize=9, color='black')

    # Thiết lập trục trái (Sales & Profit)
    ax1.set_xlabel("Sản phẩm", fontsize=11)
    ax1.set_ylabel("Giá trị ($)", fontsize=11)
    ax1.set_title("Top 5 sản phẩm bán chạy nhất (Doanh số, Lợi nhuận & Số lượng)", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(top5["Product ID"], fontsize=10)
    ax1.legend(loc="upper right")

    # Trục phụ bên phải cho Quantity (line chart)
    ax2 = ax1.twinx()
    ax2.plot(x, top5["Quantity"], color="green", marker="o", linewidth=2, label="Số lượng")
    ax2.set_ylabel("Số lượng bán", fontsize=11, color="green")
    ax2.tick_params(axis='y', labelcolor='green')

    # Thêm nhãn giá trị cho line chart
    for i, q in enumerate(top5["Quantity"]):
        ax2.text(i, q + max(top5["Quantity"])*0.02, f"{int(q)}", ha='center', va='bottom', color='green', fontsize=9)

    # Gộp legend 2 trục
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.tight_layout()
    # plt.show()
    safe_save_fig(fig, out_dir / "Top5_Products.png")

    # 8) Công ty nên đặt hàng nhiều hơn hay ít hơn những sản phẩm nào?
    product_stats = (
        df.groupby("Product ID")
        .agg({
            "Sales": "sum",
            "Profit": "sum",
            "Quantity": "sum",
            "Order Date": ["min", "max"]
        })
    )
    product_stats.columns = ["Sales", "Profit", "Quantity", "FirstOrder", "LastOrder"]

    # --- Tính tốc độ bán hàng (Velocity) ---
    product_stats["DaysActive"] = (product_stats["LastOrder"] - product_stats["FirstOrder"]).dt.days + 1
    product_stats["DaysActive"] = product_stats["DaysActive"].replace(0, 1)
    product_stats["Velocity"] = product_stats["Quantity"] / product_stats["DaysActive"]

    # --- Chuẩn hóa và tính điểm tổng hợp (Score) ---
    for col in ["Sales", "Profit", "Quantity", "Velocity"]:
        mean = product_stats[col].mean()
        std = product_stats[col].std()
        product_stats[f"{col}_z"] = (product_stats[col] - mean) / std if std > 0 else 0

    product_stats["Score"] = product_stats[[f"{c}_z" for c in ["Sales", "Profit", "Quantity", "Velocity"]]].mean(axis=1)

    # --- Chọn top 5 nên nhập thêm / ít lại ---
    top5_increase = product_stats.sort_values("Score", ascending=False).head(5).reset_index()
    top5_decrease = product_stats.sort_values("Score", ascending=True).head(5).reset_index()

    # --- Hàm vẽ biểu đồ ---
    def plot_combo(df_subset, title):
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(df_subset))
        width = 0.35

        # Vẽ Sales và Profit (cột)
        b1 = ax.bar(x - width/2, df_subset["Sales"], width=width, label="Doanh số", color="#4caf50")
        b2 = ax.bar(x + width/2, df_subset["Profit"], width=width, label="Lợi nhuận", color="#2196f3")

        ax.set_ylabel("Giá trị ($)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(df_subset["Product ID"], fontsize=9, ha='right')

        # Thêm nhãn giá trị
        ax.bar_label(b1, labels=[f"{int(v):,}" for v in df_subset["Sales"]], padding=2, fontsize=8)
        ax.bar_label(b2, labels=[f"{int(v):,}" for v in df_subset["Profit"]], padding=2, fontsize=8)

        # --- Vẽ các điểm Velocity trên cùng biểu đồ ---
        ax.scatter(x, df_subset["Velocity"], color="#f57c00", s=60, label="Tốc độ bán (số lượng/ngày)", zorder=5)

        # Hiển thị giá trị Velocity ngay trên mỗi điểm
        for i, v in enumerate(df_subset["Velocity"]):
            ax.text(i, v + max(df_subset["Velocity"]) * 0.05, f"{v:.4f}", 
                    ha='center', va='bottom', fontsize=8, color='#f57c00')

        # Gộp legend cho cả cột và điểm
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper right")



        plt.tight_layout()
        # plt.show()
        return fig

    # --- Vẽ 2 biểu đồ ---
    safe_save_fig(plot_combo(top5_increase[["Product ID", "Sales", "Profit", "Velocity"]],
                "Top 5 sản phẩm nên nhập thêm"),
                out_dir / "top5_should_increase.png")

    safe_save_fig(plot_combo(top5_decrease[["Product ID", "Sales", "Profit", "Velocity"]],
                "Top 5 sản phẩm nên nhập ít lại"),
                out_dir / "top5_should_decrease.png")

if __name__ == "__main__":
    main()