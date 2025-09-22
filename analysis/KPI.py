# ...existing code...
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

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

def make_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, pd.Interval) or isinstance(obj, pd.Period):
        return str(obj)
    try:
        if hasattr(obj, "item"):
            return make_serializable(obj.item())
    except Exception:
        pass
    return obj

def main():
    project = Path(__file__).resolve().parents[2]
    inp = project / "project/data/sample_2.csv"
    if not inp.exists():
        print(f"Không tìm thấy file: {inp}", file=sys.stderr)
        sys.exit(1)

    df = read_csv_fallback(inp)
    df.columns = [str(c).strip() for c in df.columns]

    # detect cols
    sales_col = find_col(df.columns, ["Sales","sales","Revenue","Amount"])
    profit_col = find_col(df.columns, ["Profit","profit"])
    qty_col = find_col(df.columns, ["Quantity","Qty","Order Quantity"])
    cust_col = find_col(df.columns, ["Customer ID","CustomerID","Customer"])
    order_date_col = find_col(df.columns, ["Order Date","OrderDate","Order_Date","Date"])
    ship_date_col = find_col(df.columns, ["Ship Date","ShipDate","Ship_Date"])
    region_col = find_col(df.columns, ["Region","region"])
    segment_col = find_col(df.columns, ["Segment","segment"])
    product_col = find_col(df.columns, ["Product Name","Product","Product ID"])

    if sales_col is None:
        print("Không tìm thấy cột Sales.", file=sys.stderr)
        sys.exit(1)

    # normalize numerics
    df[sales_col] = df[sales_col].astype(str).str.replace(r'[\$,]', '', regex=True)
    df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce').fillna(0.0)
    if profit_col:
        df[profit_col] = pd.to_numeric(df[profit_col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0.0)
    if qty_col:
        df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)

    # parse dates and create year_month
    if order_date_col:
        df[order_date_col] = pd.to_datetime(df[order_date_col], errors='coerce', dayfirst=False)
    if ship_date_col:
        df[ship_date_col] = pd.to_datetime(df[ship_date_col], errors='coerce', dayfirst=False)
    if order_date_col:
        df["year_month"] = df[order_date_col].dt.to_period("M").astype(str)
    else:
        df["year_month"] = "unknown"

    # KPI calculations
    total_transactions = int(len(df))
    total_revenue = float(df[sales_col].sum())
    unique_customers = int(df[cust_col].nunique()) if cust_col else None
    orders_count = int(df["Order ID"].nunique()) if "Order ID" in df.columns else total_transactions
    aov = float(total_revenue / orders_count) if orders_count>0 else 0.0

    monthly_rev = df.groupby("year_month")[sales_col].sum().sort_index()
    avg_sales_per_month = float(monthly_rev.mean()) if len(monthly_rev)>0 else 0.0

    # top region by average sales per transaction (user requested region)
    top_region = None
    if region_col:
        region_stats = df.groupby(region_col)[sales_col].agg(sum="sum", mean="mean", count="count").reset_index().sort_values("mean", ascending=False)
        if not region_stats.empty:
            top_region = region_stats.iloc[0].to_dict()

    # profits by segment
    profits_by_segment = None
    if profit_col and segment_col:
        seg = df.groupby(segment_col)[profit_col].sum().reset_index().sort_values(by=profit_col, ascending=False)
        profits_by_segment = seg.to_dict(orient="records")

    # shipping time analysis
    ship_analysis = {}
    if order_date_col and ship_date_col:
        df["ship_days"] = (df[ship_date_col] - df[order_date_col]).dt.days
        # sanitize: remove negative large errors but keep negative to inspect
        ship_analysis["avg_ship_days"] = float(df["ship_days"].dropna().mean())
        ship_analysis["median_ship_days"] = float(df["ship_days"].dropna().median())
        ship_analysis["ship_days_std"] = float(df["ship_days"].dropna().std())
        # buckets
        bins = [-999,0,1,3,7,30,9999]
        labels = ["<=0","1","2-3","4-7","8-30",">30"]
        df["ship_bucket"] = pd.cut(df["ship_days"], bins=bins, labels=labels)
        bucket_stats = df.groupby("ship_bucket")[sales_col].agg(orders="count", avg_sales="mean").reset_index()
        ship_analysis["bucket_stats"] = {str(row["ship_bucket"]): {"orders": int(row["orders"]), "avg_sales": float(row["avg_sales"] if not np.isnan(row["avg_sales"]) else 0.0)} for _, row in bucket_stats.iterrows()}
        # correlation ship_days vs sales
        try:
            corr = float(df[["ship_days", sales_col]].dropna().corr().iloc[0,1])
        except Exception:
            corr = None
        ship_analysis["ship_sales_correlation"] = corr

    # top products
    top_products = []
    if product_col:
        prod = df.groupby(product_col).agg(total_sales=(sales_col,"sum"), total_qty=(qty_col,"sum") if qty_col else (sales_col,"count"))
        top_products = prod.sort_values("total_sales", ascending=False).head(20).reset_index().to_dict(orient="records")

    # reorder recommendations (heuristic)
    reorder = {"more": [], "less": []}
    if product_col:
        prod_by_sales = df.groupby(product_col)[sales_col].sum().sort_values(ascending=False)
        reorder["more"] = prod_by_sales.head(10).index.tolist()
        reorder["less"] = prod_by_sales.tail(10).index.tolist()

    # customer LTV / CAC hint (simple)
    cac = {}
    if cust_col:
        cust_total = df.groupby(cust_col)[sales_col].sum()
        avg_ltv = float(cust_total.mean()) if len(cust_total)>0 else 0.0
        recommended_max_cac = round(0.3 * avg_ltv, 2)
        cac = {"avg_ltv": avg_ltv, "recommended_max_cac_per_new_customer": recommended_max_cac}

    summary = {
        "total_transactions": total_transactions,
        "orders_count": orders_count,
        "unique_customers": unique_customers,
        "total_revenue": total_revenue,
        "aov": aov,
        "avg_sales_per_month": avg_sales_per_month,
        "top_region_by_avg_sale": top_region,
        "profits_by_segment": profits_by_segment,
        "ship_analysis": ship_analysis,
        "top_products_sample": top_products[:10],
        "reorder_recommendations": {"more": reorder["more"][:10], "less": reorder["less"][:10]},
        "customer_acquisition_hint": cac
    }

    out_dir = project / "analysis_outputs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "monthly_revenue.csv").write_text(monthly_rev.reset_index().to_csv(index=False), encoding="utf-8-sig")
    with open(out_dir / "kpi_summary.json", "w", encoding="utf-8") as f:
        json.dump(make_serializable(summary), f, ensure_ascii=False, indent=2)

    # concise print
    print(f"Total transactions: {total_transactions}")
    print(f"Total revenue: {total_revenue:,.2f}")
    print(f"Unique customers: {unique_customers}")
    print(f"AOV: {aov:,.2f}")
    print(f"Avg sales / month: {avg_sales_per_month:,.2f}")
    if top_region:
        print(f"Top region by avg sale per transaction: {top_region.get(region_col)} (mean={top_region.get('mean'):.2f})")
    if ship_analysis:
        print(f"Avg ship days: {ship_analysis.get('avg_ship_days'):.2f}, corr(ship_days,sales)={ship_analysis.get('ship_sales_correlation')}")
    print("KPI summary written to:", out_dir / "kpi_summary.json")

if __name__ == "__main__":
    main()
# ...existing code...