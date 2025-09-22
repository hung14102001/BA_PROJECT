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
    # Recursively convert numpy / pandas / Interval / Timestamp types to native Python types
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
    if isinstance(obj, pd.Interval):
        return str(obj)
    if isinstance(obj, pd.Period):
        return str(obj)
    # pandas scalar types
    try:
        if hasattr(obj, "item"):
            return make_serializable(obj.item())
    except Exception:
        pass
    return obj

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
    qty_col = find_col(df.columns, ["Quantity", "Qty", "Order Quantity"])
    cust_col = find_col(df.columns, ["Customer ID", "CustomerID", "Customer"])
    date_col = find_col(df.columns, ["Order Date", "OrderDate", "Date", "Ship Date"])
    market_col = find_col(df.columns, ["Country", "Market", "State", "Region"])
    segment_col = find_col(df.columns, ["Segment", "segment"])
    product_col = find_col(df.columns, ["Product Name", "Product", "Product ID"])

    if sales_col is None:
        print("Không tìm thấy cột Sales.", file=sys.stderr)
        sys.exit(1)

    # Normalize numeric columns
    df[sales_col] = df[sales_col].astype(str).str.replace(r'[\$,]', '', regex=True)
    df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce').fillna(0.0)
    if profit_col:
        df[profit_col] = pd.to_numeric(df[profit_col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0.0)
    if qty_col:
        df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)

    # Parse dates
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        df["_date_"] = pd.to_datetime("1970-01-01")
        date_col = "_date_"
    df["year_month"] = df[date_col].dt.to_period("M").astype(str)

    # Q1: total number of sales (transactions) and total sales amount
    total_transactions = int(len(df))
    total_sales_amount = float(df[sales_col].sum())

    # Q2 & Q3: monthly revenue and average sales per month
    monthly_rev = df.groupby("year_month")[sales_col].sum().sort_index()
    avg_sales_per_month = float(monthly_rev.mean()) if len(monthly_rev) > 0 else 0.0

    # Q4: key demographics (age, membership if present)
    demographics = {}
    if "age" in df.columns:
        ages = df["age"].dropna().astype(int)
        demographics["count"] = int(len(ages))
        demographics["mean"] = float(ages.mean()) if len(ages)>0 else None
        demographics["median"] = float(ages.median()) if len(ages)>0 else None
        # convert Interval keys to strings to be JSON-serializable
        age_groups_series = ages.groupby(pd.cut(ages, [18,25,35,45,55,65,100])).size()
        demographics["age_groups"] = {str(k): int(v) for k, v in age_groups_series.items()}
    if "membership" in df.columns:
        demographics["membership_counts"] = df["membership"].fillna("UNKNOWN").value_counts().to_dict()

    # Q5: which market generated the most sales on average (per transaction)
    top_market = None
    if market_col:
        market_stats = df.groupby(market_col)[sales_col].agg(["sum","mean","count"]).reset_index().sort_values("mean", ascending=False)
        if not market_stats.empty:
            top_market = market_stats.iloc[0].to_dict()

    # Q6: profits by segment
    profits_by_segment = None
    if profit_col and segment_col:
        seg = df.groupby(segment_col)[profit_col].sum().reset_index().sort_values(profit_col, ascending=False)
        profits_by_segment = seg.to_dict(orient="records")

    # Q7: best and worst selling periods (by month revenue)
    best_period = {"period": None, "revenue": None}
    worst_period = {"period": None, "revenue": None}
    if len(monthly_rev) > 0:
        best_period["period"] = monthly_rev.idxmax(); best_period["revenue"] = float(monthly_rev.max())
        worst_period["period"] = monthly_rev.idxmin(); worst_period["revenue"] = float(monthly_rev.min())

    # Q8: which products sell best
    top_products = []
    if product_col:
        agg = df.groupby(product_col).agg(total_sales=(sales_col,"sum"), total_qty=(qty_col,"sum") if qty_col else (sales_col,"count"))
        top_products = agg.sort_values("total_sales", ascending=False).head(20).reset_index().to_dict(orient="records")

    # Q9: order more/less recommendations (simple heuristic)
    reorder = {"more": [], "less": []}
    if product_col:
        prod_by_sales = df.groupby(product_col)[sales_col].sum().sort_values(ascending=False)
        reorder["more"] = prod_by_sales.head(10).index.tolist()
        reorder["less"] = prod_by_sales.tail(10).index.tolist()

    # Q10: marketing adjustments for VIP vs less-engaged
    marketing = {
        "VIP": "Tăng ưu đãi cá nhân hoá, chương trình giữ chân (loyalty), cross-sell sản phẩm lợi nhuận cao, chăm sóc kênh trực tiếp (email/SMS).",
        "Less-engaged": "Chạy chiến dịch tái tương tác: ưu đãi giới hạn, A/B testing subject lines, retargeting ads, khảo sát ngắn để tìm rào cản."
    }

    # Q11: should acquire new customers & budget suggestion (simple LTV heuristic)
    cac = {}
    if cust_col:
        cust_total = df.groupby(cust_col)[sales_col].sum()
        avg_ltv = float(cust_total.mean()) if len(cust_total)>0 else 0.0
        recommended_max_cac = round(0.3 * avg_ltv, 2)  # 30% of avg LTV
        cac = {"avg_ltv": avg_ltv, "recommended_max_cac_per_new_customer": recommended_max_cac}
        cac["example_budget_for_100_new"] = round(100 * recommended_max_cac, 2)

    # Save outputs
    out_dir = project / "analysis_outputs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "monthly_revenue.csv").write_text(monthly_rev.reset_index().to_csv(index=False), encoding="utf-8-sig")
    summary = {
        "total_transactions": total_transactions,
        "total_sales_amount": total_sales_amount,
        "avg_sales_per_month": avg_sales_per_month,
        "demographics": demographics,
        "top_market_by_avg_sale": top_market,
        "profits_by_segment": profits_by_segment,
        "best_period": best_period,
        "worst_period": worst_period,
        "top_products": top_products,
        "reorder_recommendations": reorder,
        "marketing_recommendations": marketing,
        "customer_acquisition": cac
    }

    # Ensure summary is JSON serializable
    with open(out_dir / "sample_age_summary.json", "w", encoding="utf-8") as f:
        json.dump(make_serializable(summary), f, ensure_ascii=False, indent=2)

    # Print concise answers
    print(f"- Total number of sales (transactions): {total_transactions}")
    print(f"- Total sales amount: {total_sales_amount:,.2f}")
    print(f"- Average sales per month: {avg_sales_per_month:,.2f}")
    print(f"- Monthly revenue file: {out_dir / 'monthly_revenue.csv'}")
    print(f"- Key demographics (age/membership): {list(demographics.keys())}")
    if top_market:
        print(f"- Market with highest avg sale per transaction: {top_market.get(market_col)} (mean={top_market.get('mean'):.2f})")
    if profits_by_segment:
        print(f"- Profits by segment saved (top shown): {profits_by_segment[:5]}")
    print(f"- Best selling period: {best_period}")
    print(f"- Worst selling period: {worst_period}")
    print(f"- Top products sample (top 5): {top_products[:5]}")
    print(f"- Reorder recommendations (more/less): {reorder['more'][:5]} / {reorder['less'][:5]}")
    print(f"- Marketing recs & CAC saved to: {out_dir / 'sample_age_summary.json'}")

if __name__ == "__main__":
    main()