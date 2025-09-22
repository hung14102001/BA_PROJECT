import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
import sys

try:
    import seaborn as sns
    sns.set(style="whitegrid")
except Exception:
    sns = None

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
    project = Path(__file__).resolve().parents[2]
    inp = project / "project/data/sample_age.csv"
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
    df[sales_col] = df[sales_col].astype(str).str.replace(r'[\$,]', '', regex=True)
    df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce').fillna(0.0)
    if profit_col:
        df[profit_col] = pd.to_numeric(df[profit_col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0.0)
    if qty_col:
        df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)

    # dates
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        df["_date_"] = pd.to_datetime("1970-01-01")
        date_col = "_date_"
    df["year_month"] = df[date_col].dt.to_period("M").astype(str)

    out_dir = project / "analysis_outputs"
    out_dir.mkdir(exist_ok=True)

    # 1) Monthly revenue line chart
    monthly_rev = df.groupby("year_month")[sales_col].sum().sort_index()
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(monthly_rev.index, monthly_rev.values, marker="o", linewidth=1)
    ax.set_title("Monthly Revenue")
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Revenue")
    ax.tick_params(axis="x", rotation=45)
    safe_save_fig(fig, out_dir / "monthly_revenue.png")

    # 2) Sales distribution (per transaction)
    fig, ax = plt.subplots(figsize=(8,4))
    if sns:
        sns.histplot(df[sales_col], bins=80, kde=False, ax=ax)
    else:
        ax.hist(df[sales_col].dropna(), bins=80)
    ax.set_title("Distribution of Sales per Transaction")
    ax.set_xlabel("Sales")
    ax.set_ylabel("Count")
    safe_save_fig(fig, out_dir / "sales_distribution.png")

    # 3) Age distribution & avg sales by age group (if age present)
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        fig, axes = plt.subplots(1,2, figsize=(12,4))
        if sns:
            sns.histplot(df["age"].dropna(), bins=20, ax=axes[0])
        else:
            axes[0].hist(df["age"].dropna(), bins=20)
        axes[0].set_title("Age Distribution")
        axes[0].set_xlabel("Age")
        axes[0].set_ylabel("Count")

        age_bins = [18,25,35,45,55,65,100]
        df["age_group"] = pd.cut(df["age"], age_bins)
        avg_by_age = df.groupby("age_group")[sales_col].mean().dropna()
        avg_by_age.index = avg_by_age.index.astype(str)
        axes[1].bar(avg_by_age.index, avg_by_age.values)
        axes[1].set_title("Avg Sales by Age Group")
        axes[1].set_xlabel("Age Group")
        axes[1].set_ylabel("Avg Sales")
        axes[1].tick_params(axis="x", rotation=45)
        safe_save_fig(fig, out_dir / "age_distribution_and_avg_sales.png")

    # 4) Membership counts & avg sales by membership
    if membership_col:
        fig, axes = plt.subplots(1,2, figsize=(10,4))
        membership_counts = df[membership_col].fillna("UNKNOWN").value_counts()
        axes[0].bar(membership_counts.index.astype(str), membership_counts.values)
        axes[0].set_title("Membership Counts")
        axes[0].tick_params(axis="x", rotation=45)

        avg_mem = df.groupby(membership_col)[sales_col].mean().dropna()
        axes[1].bar(avg_mem.index.astype(str), avg_mem.values)
        axes[1].set_title("Avg Sales by Membership")
        axes[1].tick_params(axis="x", rotation=45)
        safe_save_fig(fig, out_dir / "membership_charts.png")

    # 5) Top products bar chart
    if product_col:
        prod_agg = df.groupby(product_col)[sales_col].agg(total_sales="sum", count="count")
        top_prod = prod_agg.sort_values("total_sales", ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(10,6))
        ax.barh(top_prod.index.astype(str)[::-1], top_prod["total_sales"].values[::-1])
        ax.set_title("Top 20 Products by Sales")
        ax.set_xlabel("Total Sales")
        safe_save_fig(fig, out_dir / "top_products.png")

    # 6) Profits by segment
    if profit_col and segment_col:
        seg = df.groupby(segment_col)[profit_col].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(seg.index.astype(str), seg.values)
        ax.set_title("Profits by Segment")
        ax.set_ylabel("Profit")
        ax.tick_params(axis="x", rotation=45)
        safe_save_fig(fig, out_dir / "profits_by_segment.png")

    # 7) Market (country) avg sales per transaction (top 10)
    if market_col:
        market_stats = df.groupby(market_col)[sales_col].agg(mean="mean", sum="sum", count="count")
        market_top = market_stats.sort_values("mean", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.bar(market_top.index.astype(str), market_top["mean"].values)
        ax.set_title("Top 10 Markets by Avg Sales per Transaction")
        ax.set_ylabel("Avg Sales")
        ax.tick_params(axis="x", rotation=45)
        safe_save_fig(fig, out_dir / "top_markets_avg_sales.png")


    # # ...existing code...
    # -    # 7) Market (country) avg sales per transaction (top 10)
    # -    if market_col:
    # -        market_stats = df.groupby(market_col)[sales_col].agg(mean="mean", sum="sum", count="count")
    # -        market_top = market_stats.sort_values("mean", ascending=False).head(10)
    # -        fig, ax = plt.subplots(figsize=(10,5))
    # -        ax.bar(market_top.index.astype(str), market_top["mean"].values)
    # -        ax.set_title("Top 10 Markets by Avg Sales per Transaction")
    # -        ax.set_ylabel("Avg Sales")
    # -        ax.tick_params(axis="x", rotation=45)
    # -        safe_save_fig(fig, out_dir / "top_markets_avg_sales.png")
        # 7) Market (state/province) avg sales per transaction (top 10)
        # ưu tiên theo state_col nếu có, ngược lại dùng market_col
        location_col = state_col if state_col else market_col
        if location_col:
            loc_stats = df.groupby(location_col)[sales_col].agg(mean="mean", sum="sum", count="count")
            loc_top = loc_stats.sort_values("mean", ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(10,5))
            ax.bar(loc_top.index.astype(str), loc_top["mean"].values)
            title_col = "State/Province" if state_col else "Market"
            ax.set_title(f"Top 10 {title_col} by Avg Sales per Transaction")
            ax.set_ylabel("Avg Sales")
            ax.tick_params(axis="x", rotation=45)
            file_safe = title_col.lower().replace("/", "_").replace(" ", "_")
            safe_save_fig(fig, out_dir / f"top_{file_safe}_avg_sales.png")
    # ...existing code...

    # 8) Heatmap: monthly revenue (years x months) if multi-year
    try:
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
            safe_save_fig(fig, out_dir / "monthly_revenue_heatmap.png")
    except Exception:
        pass

    # 9) Save a short text summary
    summary = {
        "total_transactions": int(len(df)),
        "total_sales": float(df[sales_col].sum()),
        "avg_sales_per_month": float(monthly_rev.mean()) if len(monthly_rev)>0 else 0.0,
    }
    with open(out_dir / "visual_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Đã sinh biểu đồ vào:", out_dir)

if __name__ == "__main__":
    main()