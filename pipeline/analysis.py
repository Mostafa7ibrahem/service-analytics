import pandas as pd
import numpy as np
from typing import Dict, Any

def _c(df, name):
    for c in df.columns:
        if c.lower().strip().replace(" ", "_") == name.lower().strip().replace(" ", "_"):
            return c
    return None

# ─────────────────────────────────────────────
#  PAGE 1 — Executive Dashboard
# ─────────────────────────────────────────────

def exec_kpis(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    k = {}
    orders = dfs.get("orders")
    ratings = dfs.get("ratings")
    users = dfs.get("users")
    workers = dfs.get("workers")

    k["total_orders"] = len(orders) if orders is not None else 0

    # Active Workers (workers with completed orders) + Active Rate
    k["active_workers"] = 0
    k["active_workers_rate"] = 0
    if orders is not None and workers is not None:
        st = _c(orders, "status")
        wu = _c(orders, "worker_username")
        if st and wu:
            completed_workers = orders[orders[st].str.upper() == "COMPLETED"][wu].nunique()
            total_workers = len(workers)
            k["active_workers"] = int(completed_workers)
            k["active_workers_rate"] = round(completed_workers / total_workers * 100, 1) if total_workers else 0

    # Avg Worker Rating
    k["avg_worker_rating"] = 0
    if ratings is not None and len(ratings):
        st_col = _c(ratings, "stars")
        if st_col:
            k["avg_worker_rating"] = round(float(ratings[st_col].mean()), 2)

    # Completion Rate & Cancellation Rate
    k["completion_rate"] = 0
    k["cancellation_rate"] = 0
    if orders is not None and len(orders):
        st = _c(orders, "status")
        if st:
            total = len(orders)
            completed = int((orders[st].str.upper() == "COMPLETED").sum())
            cancelled = int((orders[st].str.upper() == "CANCELLED").sum())
            k["completion_rate"] = round(completed / total * 100, 1)
            k["cancellation_rate"] = round(cancelled / total * 100, 1)

    # Customer Growth Rate
    k["customer_growth_rate"] = 0
    if users is not None and len(users):
        r = _c(users, "role")
        dj = _c(users, "date_joined")
        if r and dj:
            clients = users[users[r].str.lower() == "client"].copy()
            if len(clients) > 0:
                clients["_month"] = clients[dj].dt.to_period("M").astype(str)
                monthly = clients.groupby("_month").size()
                if len(monthly) >= 2:
                    cur = monthly.iloc[-1]
                    prev = monthly.iloc[-2]
                    k["customer_growth_rate"] = round(((cur - prev) / prev) * 100, 1) if prev else 0

    return k


def exec_insights(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    ins = {}
    orders = dfs.get("orders")
    ratings = dfs.get("ratings")
    users = dfs.get("users")

    # ── 1. Best Services (bar chart - orders by category) ──
    if orders is not None and len(orders):
        cat = _c(orders, "category")
        if cat:
            best = orders[cat].value_counts().reset_index()
            best.columns = ["service", "orders"]
            best = best.head(8)
            ins["best_services"] = best.to_dict("records")

    # ── 2. Province Stats — clients & workers per governorate ──
    if users is not None and len(users):
        gov = _c(users, "governorate")
        role = _c(users, "role")
        if gov and role:
            prov = users.groupby([gov, role]).size().unstack(fill_value=0).reset_index()
            prov.columns.name = None
            prov["Total"] = prov.iloc[:, 1:].sum(axis=1)
            col_map = {}
            for c in prov.columns:
                if isinstance(c, str):
                    col_map[c] = c.title()
            if "Client" in prov.columns:
                pass
            prov = prov.sort_values("Total", ascending=False)
            ins["province_stats"] = prov.to_dict("records")
            ins["province_cols"] = prov.columns.tolist()

    # ── 3. Status Distribution (pie) ──
    if orders is not None and len(orders):
        st = _c(orders, "status")
        if st:
            labels = {"completed": "✅ Completed", "cancelled": "❌ Cancelled",
                      "pending": "⏳ Pending", "in_progress": "🔄 In Progress"}
            counts = orders[st].value_counts()
            renamed = {}
            for k, v in counts.items():
                renamed[labels.get(k.lower(), k.title())] = int(v)
            ins["process_pie"] = renamed

    # ── 4. Customer Satisfaction Trend (line with target) ──
    if ratings is not None and len(ratings):
        rc = _c(ratings, "created_at")
        rr = _c(ratings, "stars")
        if rc and rr:
            r = ratings.copy()
            r["_month"] = r[rc].dt.to_period("M").astype(str)
            trend = r.groupby("_month")[rr].mean().reset_index()
            trend.columns = ["month", "avg_rating"]
            ins["satisfaction_trend"] = trend.to_dict("records")

    return ins


# ─────────────────────────────────────────────
#  PAGE 2 — Workforce & Services
# ─────────────────────────────────────────────

def workforce_kpis(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    k = {}
    users = dfs.get("users")
    orders = dfs.get("orders")
    workers = dfs.get("workers")
    ratings = dfs.get("ratings")

    # ── 1. Total Workers ──
    k["total_workers"] = len(workers) if workers is not None else 0
    total_workers = k["total_workers"]

    # ── 2. Active Workers Rate ──
    k["active_workers_rate"] = 0
    if orders is not None and total_workers:
        wu = _c(orders, "worker_username")
        st = _c(orders, "status")
        if wu and st:
            active = int(orders[orders[st].str.upper() == "COMPLETED"][wu].nunique())
            k["active_workers_rate"] = round(active / total_workers * 100, 1) if total_workers else 0

    # ── 3. Avg Orders per Worker ──
    k["avg_orders_per_worker"] = round(len(orders) / total_workers, 1) if orders is not None and total_workers else 0

    # ── 4. Avg Worker Rating ──
    if ratings is not None and len(ratings):
        rr = _c(ratings, "stars")
        k["avg_worker_rating"] = round(float(ratings[rr].mean()), 2) if rr else 0
    else:
        k["avg_worker_rating"] = 0

    # ── 5. Top Service Category ──
    if orders is not None and len(orders):
        cat = _c(orders, "category")
        if cat:
            k["top_category"] = orders[cat].value_counts().idxmax()
        else:
            k["top_category"] = "N/A"
    else:
        k["top_category"] = "N/A"

    # ── 6. Revenue per Worker (commission-based) ──
    if orders is not None and len(orders) and total_workers:
        cm = _c(orders, "commission")
        if cm:
            k["revenue_per_worker"] = round(float(orders[cm].sum()) / total_workers, 2)
        else:
            k["revenue_per_worker"] = 0
    else:
        k["revenue_per_worker"] = 0

    return k


def workforce_insights(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    ins = {}
    orders = dfs.get("orders")
    users = dfs.get("users")
    workers = dfs.get("workers")
    ratings = dfs.get("ratings")

    # ── 1. Worker Leaderboard ──
    if workers is not None and orders is not None and len(orders):
        w_uname = _c(workers, "username")
        w_name = _c(workers, "name")
        prof = _c(workers, "profession")
        exp = _c(workers, "experience_years")
        gov = _c(workers, "governorate")
        ord_wu = _c(orders, "worker_username")
        cm = _c(orders, "commission")
        if w_uname and w_name and ord_wu:
            merged = workers[[w_uname, w_name, prof, exp, gov]].copy() if prof and exp and gov else workers[[w_uname, w_name]].copy()
            merged.columns = [c if c != w_uname else "username" for c in merged.columns]
            # revenue per worker
            if cm:
                rev = orders.groupby(ord_wu)[cm].sum().reset_index()
                rev.columns = ["username", "revenue"]
                merged = merged.merge(rev, on="username", how="left").fillna(0)
            # order count per worker
            oc = orders.groupby(ord_wu).size().reset_index(name="orders").rename(columns={ord_wu: "username"})
            merged = merged.merge(oc, on="username", how="left").fillna(0)
            # avg rating per worker
            if ratings is not None and len(ratings):
                rwu = _c(ratings, "worker_username")
                rr = _c(ratings, "stars")
                if rwu and rr:
                    r_avg = ratings.groupby(rwu)[rr].mean().reset_index()
                    r_avg.columns = ["username", "avg_rating"]
                    merged = merged.merge(r_avg, on="username", how="left").fillna(0)
            # composite score
            score_cols = []
            if "revenue" in merged.columns and merged["revenue"].max() > 0:
                merged["_rev_norm"] = merged["revenue"] / merged["revenue"].max()
                score_cols.append("_rev_norm")
            if "avg_rating" in merged.columns and merged["avg_rating"].max() > 0:
                merged["_rating_norm"] = merged["avg_rating"] / 5.0
                score_cols.append("_rating_norm")
            if merged["orders"].max() > 0:
                merged["_ord_norm"] = merged["orders"] / merged["orders"].max()
                score_cols.append("_ord_norm")
            merged["_score"] = merged[score_cols].sum(axis=1) if score_cols else 0
            top10 = merged.sort_values("_score", ascending=False).head(10)
            bottom10 = merged[merged["orders"] > 0].sort_values("_score", ascending=True).head(10)
            ins["worker_leaderboard_top"] = top10.to_dict("records")
            ins["worker_leaderboard_bottom"] = bottom10.to_dict("records")
            ins["leaderboard_cols"] = [c for c in ["name", "governorate", "profession", "experience_years", "revenue", "avg_rating", "orders", "_score"]
                                        if c in top10.columns]

    # ── 2. Quality Matrix (Scatter: orders vs rating per worker) ──
    if workers is not None and orders is not None and ratings is not None:
        owu = _c(orders, "worker_username")
        rwu = _c(ratings, "worker_username")
        rr = _c(ratings, "stars")
        w_n = _c(workers, "name")
        w_u = _c(workers, "username")
        if owu and rwu and rr:
            oc = orders.groupby(owu).size().reset_index(name="orders")
            r_avg = ratings.groupby(rwu)[rr].mean().reset_index()
            r_avg.columns = ["username", "avg_rating"]
            qm = oc.merge(r_avg, left_on=owu, right_on="username", how="inner").dropna()
            if w_n and w_u and len(qm):
                w_df = workers[[w_u, w_n]].copy()
                w_df.columns = ["username", "name"]
                qm = qm.merge(w_df, on="username", how="left")
            if len(qm) > 2:
                ins["quality_matrix"] = qm.to_dict("records")

    # ── 3. Capacity vs Demand ──
    if workers is not None and orders is not None:
        prof = _c(workers, "profession")
        cat = _c(orders, "category")
        if prof and cat:
            wp = workers[prof].value_counts().reset_index()
            wp.columns = ["service", "workers"]
            oc = orders[cat].value_counts().reset_index()
            oc.columns = ["service", "orders"]
            merged = wp.merge(oc, on="service", how="outer").fillna(0)
            merged["workers"] = merged["workers"].astype(int)
            merged["orders"] = merged["orders"].astype(int)
            merged["ratio"] = merged.apply(
                lambda r: round(r["orders"] / r["workers"], 1) if r["workers"] > 0 else 0, axis=1)
            ins["capacity_vs_demand"] = merged.to_dict("records")

    # ── 4. Service Revenue & Cancellation Analysis ──
    if orders is not None and len(orders):
        cat = _c(orders, "category")
        st = _c(orders, "status")
        cm = _c(orders, "commission")
        if cat and st:
            o = orders.copy()
            total = o.groupby(cat).size().reset_index(name="total_orders")
            cancelled = o[o[st].str.upper() == "CANCELLED"].groupby(cat).size().reset_index(name="cancelled")
            svc = total.merge(cancelled, on=cat, how="left").fillna(0)
            svc["cancelled"] = svc["cancelled"].astype(int)
            svc["cancel_rate"] = round(svc["cancelled"] / svc["total_orders"] * 100, 1)
            if cm:
                rev = o.groupby(cat)[cm].sum().reset_index()
                rev.columns = [cat, "revenue"]
                svc = svc.merge(rev, on=cat, how="left")
            ins["service_revenue_cancel"] = svc.to_dict("records")

    # ── 5. Most / Least Requested Services ──
    if orders is not None and len(orders):
        cat = _c(orders, "category")
        if cat:
            oc = orders[cat].value_counts().reset_index()
            oc.columns = ["service", "orders"]
            ins["most_requested"] = oc.sort_values("orders", ascending=False).to_dict("records")
            ins["least_requested"] = oc.sort_values("orders", ascending=True).to_dict("records")

    # ── 6. Quality Trend (avg rating per month) ──
    if ratings is not None and len(ratings):
        rc = _c(ratings, "created_at")
        rr = _c(ratings, "stars")
        if rc and rr:
            r = ratings.copy()
            r["_month"] = r[rc].dt.to_period("M").astype(str)
            trend = r.groupby("_month")[rr].mean().reset_index()
            trend.columns = ["month", "avg_rating"]
            ins["quality_trend"] = trend.to_dict("records")

    # ── 7. Profession Overview ──
    try:
        if workers is not None and len(workers):
            prof = _c(workers, "profession")
            exp = _c(workers, "experience_years")
            ar = _c(workers, "average_rating")
            if prof:
                gb = workers.groupby(prof)
                overview = gb.agg(
                    worker_count=("user_id", "count"),
                    avg_experience=(exp, "mean") if exp else ("user_id", "count"),
                    avg_rating=(ar, "mean") if ar else ("user_id", "count"),
                ).reset_index()
                overview.columns = ["profession", "worker_count", "avg_experience", "avg_rating"]
                overview["avg_experience"] = overview["avg_experience"].round(1)
                overview["avg_rating"] = overview["avg_rating"].round(2)
                ins["profession_overview"] = overview.to_dict("records")
    except Exception:
        pass

    # ── 8. Worker Availability by Profession ──
    try:
        if workers is not None and len(workers):
            prof = _c(workers, "profession")
            iav = _c(workers, "is_available")
            if prof and iav is not None:
                avail = workers.groupby([prof, iav]).size().reset_index(name="count")
                avail.columns = ["profession", "is_available", "count"]
                ins["worker_availability"] = avail.to_dict("records")
    except Exception:
        pass

    # ── 9. Acceptance Rate by Profession ──
    try:
        if workers is not None and len(workers):
            prof = _c(workers, "profession")
            arate = _c(workers, "accept_rate")
            if prof and arate:
                acc = workers.groupby(prof)[arate].mean().reset_index()
                acc.columns = ["profession", "avg_accept_rate"]
                acc["avg_accept_rate"] = acc["avg_accept_rate"].round(1)
                ins["acceptance_rate"] = acc.to_dict("records")
    except Exception:
        pass

    # ── 10. Experience vs Rating ──
    try:
        if workers is not None and len(workers):
            prof = _c(workers, "profession")
            exp = _c(workers, "experience_years")
            ar = _c(workers, "average_rating")
            if prof and exp and ar:
                er = workers.groupby(prof).agg(
                    avg_experience=(exp, "mean"),
                    avg_rating=(ar, "mean"),
                ).reset_index()
                er.columns = ["profession", "avg_experience", "avg_rating"]
                er["avg_experience"] = er["avg_experience"].round(1)
                er["avg_rating"] = er["avg_rating"].round(2)
                ins["exp_vs_rating"] = er.to_dict("records")
    except Exception:
        pass

    return ins


# ─────────────────────────────────────────────
#  PAGE 3 — Customer & Order Funnel
# ─────────────────────────────────────────────

def customer_kpis(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    k = {}
    users = dfs.get("users")
    orders = dfs.get("orders")

    # ── 1. Total Clients ──
    if users is not None and len(users):
        r = _c(users, "role")
        k["total_clients"] = int((users[r].str.lower() == "client").sum()) if r else 0
    else:
        k["total_clients"] = 0
    total_clients = k["total_clients"]

    # ── 2. Repeat Customer Rate ──
    k["repeat_rate"] = 0
    k["repeat_clients"] = 0
    if orders is not None and len(orders):
        cu = _c(orders, "client_username")
        if cu:
            client_orders = orders[cu].value_counts()
            repeat = int((client_orders > 1).sum())
            k["repeat_clients"] = repeat
            k["repeat_rate"] = round(repeat / total_clients * 100, 1) if total_clients else 0

    # ── 3. Avg Orders per Client ──
    k["avg_orders_per_client"] = round(len(orders) / total_clients, 1) if orders is not None and total_clients else 0

    # ── 4. Avg Fulfillment Time (hours) ──
    k["avg_fulfillment_hours"] = 0
    if orders is not None and len(orders):
        ca = _c(orders, "created_at")
        co = _c(orders, "completed_at")
        st = _c(orders, "status")
        if ca and co and st:
            completed = orders[orders[st].str.upper() == "COMPLETED"].copy()
            if len(completed):
                diff = (completed[co] - completed[ca]).dt.total_seconds() / 3600
                k["avg_fulfillment_hours"] = round(float(diff.mean()), 1)

    # ── 5. Cancellation Rate ──
    k["cancellation_rate"] = 0
    if orders is not None and len(orders):
        st = _c(orders, "status")
        if st:
            cancelled = int((orders[st].str.upper() == "CANCELLED").sum())
            k["cancellation_rate"] = round(cancelled / len(orders) * 100, 1)

    return k


def customer_insights(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    ins = {}
    orders = dfs.get("orders")
    users = dfs.get("users")
    ratings = dfs.get("ratings")

    # ── 1. Order Funnel ──
    if orders is not None and len(orders):
        st = _c(orders, "status")
        if st:
            counts = orders[st].value_counts()
            labels_map = {
                "completed": "✅ Completed", "cancelled": "❌ Cancelled",
                "pending": "⏳ Pending", "in_progress": "🔄 In Progress",
                "rejected": "🚫 Rejected",
            }
            funnel = {}
            for k, v in counts.items():
                funnel[labels_map.get(k.lower(), k.title())] = int(v)
            ins["order_funnel"] = funnel

    # ── 2. Customer Segmentation ──
    if orders is not None and len(orders) and users is not None:
        cu = _c(orders, "client_username")
        cm = _c(orders, "commission")
        st = _c(orders, "status")
        uname_u = _c(users, "username")
        uname_n = _c(users, "name")
        role = _c(users, "role")
        if cu and uname_u and role:
            client_stats = orders.groupby(cu).agg(
                total_orders=(st, "count"),
                total_spend=(cm, "sum") if cm else (st, "count"),
                cancelled=(st, lambda x: (x.str.upper() == "CANCELLED").sum()) if st else (st, lambda x: 0)
            ).reset_index()
            client_stats.columns = ["username", "total_orders", "total_spend", "cancelled"]
            avg_spend = client_stats["total_spend"].mean()
            def segment(r):
                if r["total_orders"] >= 3 and r["total_spend"] > avg_spend:
                    return "🌟 VIP"
                if r["total_orders"] >= 2:
                    return "💎 Loyal"
                if r["cancelled"] > (r["total_orders"] - r["cancelled"]):
                    return "⚠️ At Risk"
                return "🆕 New"
            client_stats["segment"] = client_stats.apply(segment, axis=1)
            u_df = users[users[role].str.lower() == "client"][[uname_u, uname_n]].copy()
            u_df.columns = ["username", "name"]
            client_stats = client_stats.merge(u_df, on="username", how="left")
            ins["customer_segmentation"] = client_stats.to_dict("records")
            seg_counts = client_stats["segment"].value_counts().reset_index()
            seg_counts.columns = ["segment", "count"]
            ins["segment_counts"] = seg_counts.to_dict("records")

    # ── 3. Orders by Governorate ──
    if orders is not None and users is not None:
        cu = _c(orders, "client_username")
        uname_u = _c(users, "username")
        gov = _c(users, "governorate")
        if cu and uname_u and gov:
            o = orders.groupby(cu).size().reset_index(name="orders")
            o.columns = ["username", "orders"]
            u = users[[uname_u, gov]]
            u.columns = ["username", "governorate"]
            merged = o.merge(u, on="username", how="left").dropna(subset=["governorate"])
            if len(merged):
                by_gov = merged.groupby("governorate")["orders"].sum().sort_values(ascending=False).reset_index()
                ins["geo_orders"] = by_gov.to_dict("records")

    # ── 4. Peak Hours Heatmap ──
    if orders is not None and len(orders):
        ca = _c(orders, "created_at")
        if ca:
            o = orders.copy()
            o["hour"] = o[ca].dt.hour
            o["dow"] = o[ca].dt.day_name()
            heat = o.groupby(["dow", "hour"]).size().reset_index(name="orders")
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            heat["dow"] = pd.Categorical(heat["dow"], categories=dow_order, ordered=True)
            heat = heat.sort_values(["dow", "hour"])
            ins["peak_hours_heatmap"] = heat.to_dict("records")
            by_hour = o.groupby("hour").size().reset_index(name="count")
            ins["orders_by_hour"] = by_hour.to_dict("records")
            by_dow = o.groupby("dow").size().reset_index(name="count")
            by_dow["dow"] = pd.Categorical(by_dow["dow"], categories=dow_order, ordered=True)
            by_dow = by_dow.sort_values("dow")
            ins["orders_by_dow"] = by_dow.to_dict("records")

    # ── 5. Fulfillment Time Distribution ──
    if orders is not None and len(orders):
        ca = _c(orders, "created_at")
        co = _c(orders, "completed_at")
        st = _c(orders, "status")
        if ca and co and st:
            completed = orders[orders[st].str.upper() == "COMPLETED"].copy()
            if len(completed):
                hours = (completed[co] - completed[ca]).dt.total_seconds() / 3600
                ins["time_to_complete"] = hours.tolist()
                cat = _c(orders, "category")
                if cat:
                    comp2 = completed.copy()
                    comp2["hours"] = (comp2[co] - comp2[ca]).dt.total_seconds() / 3600
                    by_svc = comp2.groupby(cat)["hours"].mean().reset_index()
                    by_svc["hours"] = by_svc["hours"].round(1)
                    ins["fulfillment_by_service"] = by_svc.to_dict("records")

    # ── 6. Cancelled Orders by Category ──
    if orders is not None and len(orders):
        cat = _c(orders, "category")
        st = _c(orders, "status")
        if cat and st:
            o = orders.copy()
            total = o.groupby(cat).size().reset_index(name="total")
            cancelled = o[o[st].str.upper() == "CANCELLED"].groupby(cat).size().reset_index(name="cancelled")
            merged = total.merge(cancelled, on=cat, how="left").fillna(0)
            merged["cancelled"] = merged["cancelled"].astype(int)
            merged["cancel_rate"] = (merged["cancelled"] / merged["total"] * 100).round(1)
            ins["cancelled_by_category"] = merged.to_dict("records")

    # ── 7. Satisfaction by Service ──
    if ratings is not None and orders is not None:
        roid = _c(ratings, "order_id")
        rr = _c(ratings, "stars")
        oid = _c(orders, "id")
        cat = _c(orders, "category")
        if roid and rr and oid and cat:
            merged = ratings.merge(orders, left_on=roid, right_on=oid, suffixes=("", "_o"))
            if len(merged):
                by_svc = merged.groupby(cat)[rr].mean().reset_index()
                by_svc.columns = ["service", "avg_rating"]
                by_svc["avg_rating"] = by_svc["avg_rating"].round(2)
                ins["satisfaction_by_service"] = by_svc.to_dict("records")

    return ins


# ─────────────────────────────────────────────
#  PAGE 4 — Financial & Payment Analysis
# ─────────────────────────────────────────────

def financial_kpis(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    k = {}
    orders = dfs.get("orders")
    workers = dfs.get("workers")

    # Commission-based KPIs
    k["total_collected"] = 0
    k["total_pending"] = 0
    k["avg_transaction_value"] = 0
    k["failure_rate"] = 0
    k["avg_daily_commission"] = 0
    k["max_transaction"] = 0

    if orders is not None and len(orders):
        cm = _c(orders, "commission")
        st = _c(orders, "status")
        ca = _c(orders, "created_at")
        if cm:
            completed = orders[orders[st].str.upper() == "COMPLETED"] if st else orders
            k["total_collected"] = float(completed[cm].sum())
            k["avg_transaction_value"] = round(float(completed[cm].mean()), 2) if len(completed) else 0
            k["max_transaction"] = float(completed[cm].max()) if len(completed) else 0
            if ca:
                date_range = (orders[ca].max() - orders[ca].min()).days
                k["avg_daily_commission"] = round(float(orders[cm].sum()) / max(date_range, 1), 2)

    # Effective Rate (commission / total value proxy — using avg commission rate from workers)
    k["effective_rate"] = 0
    k["monthly_growth"] = 0
    if orders is not None and len(orders):
        cm = _c(orders, "commission")
        ca = _c(orders, "created_at")
        if cm and ca:
            o = orders.copy()
            o["_month"] = o[ca].dt.to_period("M").astype(str)
            monthly = o.groupby("_month")[cm].sum()
            if len(monthly) >= 2:
                last_val = monthly.iloc[-1]
                prev_val = monthly.iloc[-2]
                k["monthly_growth"] = round(((last_val - prev_val) / prev_val) * 100, 1) if prev_val else 0
        # effective rate estimated from workers' avg commission rate
        if workers is not None and len(workers):
            hr = _c(workers, "hourly_rate")
            mc = _c(workers, "minimum_charge")
            if hr and mc:
                avg_rate = (workers[hr].mean() / workers[mc].mean() * 100) if workers[mc].mean() else 0
                k["effective_rate"] = round(avg_rate, 2)

    return k


def financial_insights(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    ins = {}
    orders = dfs.get("orders")
    users = dfs.get("users")
    workers = dfs.get("workers")

    # ── 1. Revenue & Commission Trends ──
    if orders is not None and len(orders):
        ca = _c(orders, "created_at")
        cm = _c(orders, "commission")
        if ca and cm:
            o = orders.copy()
            o["_month"] = o[ca].dt.to_period("M").astype(str)
            monthly = o.groupby("_month")[cm].sum().reset_index()
            monthly.columns = ["month", "revenue"]
            monthly["commission"] = monthly["revenue"]
            ins["revenue_commission_trends"] = monthly.to_dict("records")

    # ── 2. Top Earning Workers ──
    if orders is not None and len(orders):
        wu = _c(orders, "worker_username")
        cm = _c(orders, "commission")
        if wu and cm:
            top_w = orders.groupby(wu)[cm].sum().reset_index()
            top_w.columns = ["username", "total_commission"]
            top_w = top_w.sort_values("total_commission", ascending=False).head(5)
            if workers is not None:
                w_u = _c(workers, "username")
                w_n = _c(workers, "name")
                if w_u and w_n:
                    w_df = workers[[w_u, w_n]].copy()
                    w_df.columns = ["username", "name"]
                    top_w = top_w.merge(w_df, on="username", how="left")
            ins["top_earning_workers"] = top_w.to_dict("records")

    # ── 3. Revenue by Service Category ──
    if orders is not None and len(orders):
        cat = _c(orders, "category")
        cm = _c(orders, "commission")
        if cat and cm:
            by_svc = orders.groupby(cat)[cm].sum().reset_index().sort_values(by=cm, ascending=False)
            by_svc.columns = ["service", "revenue"]
            by_svc["commission"] = by_svc["revenue"]
            by_svc["revenue"] = by_svc["revenue"].round(2)
            by_svc["commission"] = by_svc["commission"].round(2)
            ins["revenue_by_service"] = by_svc.to_dict("records")

    # ── 4. Revenue by Governorate ──
    if orders is not None and users is not None:
        cu = _c(orders, "client_username")
        cm = _c(orders, "commission")
        uname_u = _c(users, "username")
        gov = _c(users, "governorate")
        if cu and cm and uname_u and gov:
            o_rev = orders.groupby(cu)[cm].sum().reset_index()
            o_rev.columns = ["username", "revenue"]
            u_df = users[[uname_u, gov]].dropna(subset=[gov])
            u_df.columns = ["username", "governorate"]
            merged = o_rev.merge(u_df, on="username", how="left")
            if len(merged):
                by_gov = merged.groupby("governorate")["revenue"].sum().sort_values(ascending=False).reset_index()
                by_gov["revenue"] = by_gov["revenue"].round(2)
                ins["revenue_by_governorate"] = by_gov.to_dict("records")

    # ── 5. Commission by Day of Week ──
    if orders is not None and len(orders):
        ca = _c(orders, "created_at")
        cm = _c(orders, "commission")
        if ca and cm:
            o = orders.copy()
            o["dow"] = o[ca].dt.day_name()
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            by_dow = o.groupby("dow")[cm].sum().reset_index()
            by_dow.columns = ["day", "commission"]
            by_dow["day"] = pd.Categorical(by_dow["day"], categories=dow_order, ordered=True)
            by_dow = by_dow.sort_values("day")
            by_dow["commission"] = by_dow["commission"].round(2)
            ins["commission_by_dow"] = by_dow.to_dict("records")

    return ins


# ─────────────────────────────────────────────
#  Legacy compatibility
# ─────────────────────────────────────────────

def compute_kpis(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    return exec_kpis(dfs)


def compute_insights(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    return exec_insights(dfs)


def compute_correlations(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    res = {}
    orders = dfs.get("orders")
    if orders is not None and len(orders) > 1:
        n = orders.select_dtypes(include=[np.number])
        if len(n.columns) > 1:
            c = n.corr()
            pairs = {}
            for i in c.columns:
                for j in c.index:
                    if i != j and f"{j}|{i}" not in pairs:
                        pairs[f"{i}|{j}"] = round(float(c.loc[j, i]), 3)
            res["order_corr"] = pairs
    ratings = dfs.get("ratings")
    if ratings is not None and orders is not None and len(ratings):
        m = ratings.merge(orders, left_on="order_id", right_on="id", suffixes=("", "_o"))
        n = m.select_dtypes(include=[np.number])
        if len(n.columns) > 1:
            c = n.corr()
            rr = _c(ratings, "stars")
            if rr and rr in c.columns:
                res["rating_corr"] = {k: round(float(v), 3) for k, v in c[rr].drop(rr).items()}
    return res
