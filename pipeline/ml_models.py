import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

from .preprocessing import clean_for_ml


def _c(df, name):
    for c in df.columns:
        if c.lower().strip() == name.lower().strip():
            return c
    return None


def _build(df, dfs):
    workers = dfs.get("workers")
    cats = dfs.get("categories")
    cat_col = _c(df, "category")
    if cats is not None and not cats.empty and cat_col:
        n_col = _c(cats, "name")
        if n_col:
            le = LabelEncoder()
            df["_cat_enc"] = le.fit_transform(df[cat_col].fillna("Unknown"))
    if workers is not None and len(workers):
        cols_map = {
            "username": "username",
            "experience_years": "exp_years",
            "average_rating": "avg_rating",
            "completed_jobs": "completed_jobs",
            "hourly_rate": "hourly_rate",
            "minimum_charge": "min_charge",
            "accept_rate": "accept_rate",
            "is_available": "is_available",
        }
        wf_cols = []
        rename = {}
        for orig, alias in cols_map.items():
            col = _c(workers, orig)
            if col:
                wf_cols.append(col)
                rename[col] = alias
        if "username" in rename:
            wf = workers[wf_cols].copy().rename(columns=rename)
            wu = _c(df, "worker_username")
            if wu:
                df = df.merge(wf, left_on=wu, right_on="username", how="left")
    return df


def train_models(dfs):
    res = {}
    m1 = _commission_predictor(dfs)
    if m1: res["commission_predictor"] = m1
    m2 = _completion_classifier(dfs)
    if m2: res["completion_classifier"] = m2
    m3 = _rating_predictor(dfs)
    if m3: res["rating_predictor"] = m3
    m4 = _churn_predictor(dfs)
    if m4: res["churn_predictor"] = m4
    m5 = _worker_recommender(dfs)
    if m5: res["worker_recommender"] = m5
    m6 = _worker_clustering(dfs)
    if m6: res["worker_clustering"] = m6
    m7 = generate_recommendations(dfs, res)
    if m7: res["recommendations"] = m7
    m8 = demand_forecast(dfs)
    if m8: res["demand_forecast"] = m8
    m9 = optimal_pricing_zones(dfs)
    if m9: res["optimal_pricing"] = m9
    m10 = geographic_opportunity(dfs)
    if m10: res["geographic_opportunity"] = m10
    return res


def _get_features(d):
    return [c for c in [
        "_cat_enc", "exp_years", "avg_rating", "completed_jobs",
        "hourly_rate", "min_charge", "accept_rate",
    ] if c in d.columns]


def _commission_predictor(dfs):
    try:
        o = dfs.get("orders")
        if o is None or len(o) < 30: return None
        s = _c(o, "status")
        if not s: return None
        d = o[o[s].str.upper().isin(["COMPLETED", "CANCELLED"])].copy()
        if len(d) < 30: return None
        d = _build(d, dfs)
        pr = _c(o, "commission")
        if not pr: return None
        feats = _get_features(d)
        if not feats: return None
        cleaned = clean_for_ml(d[feats + [pr]], target_col=pr)
        d_clean = cleaned["df"]
        if len(d_clean) < 30: return None
        X = d_clean[feats].values; y = d_clean[pr].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        mod = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1).fit(Xtr, ytr)
        yp = mod.predict(Xte)
        return {"type": "regression", "target": "commission",
                "rmse": round(float(np.sqrt(mean_squared_error(yte, yp))), 2),
                "r2": round(float(r2_score(yte, yp)), 3),
                "train": len(Xtr), "test": len(Xte),
                "fi": {f: round(v, 4) for f, v in zip(feats, mod.feature_importances_)},
                "model": mod}
    except Exception: return None


def _completion_classifier(dfs):
    try:
        o = dfs.get("orders")
        if o is None or len(o) < 30: return None
        s = _c(o, "status")
        if not s: return None
        d = o[o[s].str.upper().isin(["COMPLETED", "CANCELLED"])].copy()
        if len(d) < 30: return None
        d["target"] = (d[s].str.upper() == "COMPLETED").astype(int)
        d = _build(d, dfs)
        pr = _c(o, "commission")
        feats = [pr] if pr else []
        feats += [f for f in _get_features(d) if f != "_cat_enc"]
        feats = [f for f in feats if f in d.columns]
        cat_enc = _get_features(d)
        if "_cat_enc" in d.columns:
            feats.append("_cat_enc")
        if not feats: return None
        cleaned = clean_for_ml(d[feats + ["target"]], target_col="target")
        d_clean = cleaned["df"]
        if len(d_clean) < 30: return None
        X = d_clean[feats].values; y = d_clean["target"].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        mod = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1).fit(Xtr, ytr)
        yp = mod.predict(Xte)
        cr = classification_report(yte, yp, output_dict=True, zero_division=0)
        return {"type": "classification", "target": "order_completion",
                "accuracy": round(float(accuracy_score(yte, yp)), 3),
                "f1": round(float(cr["1"]["f1-score"]), 3),
                "train": len(Xtr), "test": len(Xte),
                "fi": {f: round(v, 4) for f, v in zip(feats, mod.feature_importances_)},
                "model": mod}
    except Exception: return None


def _rating_predictor(dfs):
    try:
        r = dfs.get("ratings")
        if r is None or len(r) < 30: return None
        o = dfs.get("orders")
        if o is None: return None
        d = r.copy()
        wu = _c(r, "worker_username")
        cu = _c(r, "client_username")
        star = _c(r, "stars")
        if not star: return None
        d = _build(d, dfs)
        pr = _c(o, "commission")
        o_agg = o.groupby("worker_username").agg(
            avg_commission=("commission", "mean"),
            order_count=("id", "count"),
        ).reset_index()
        o_agg.columns = ["username", "avg_commission", "worker_order_count"]
        if wu:
            d = d.merge(o_agg, left_on=wu, right_on="username", how="left")
        feats = [c for c in [
            "avg_commission", "worker_order_count", "exp_years", "avg_rating",
            "completed_jobs", "hourly_rate", "min_charge", "accept_rate",
        ] if c in d.columns]
        if not feats: return None
        cleaned = clean_for_ml(d[feats + [star]], target_col=star)
        d_clean = cleaned["df"]
        if len(d_clean) < 30: return None
        X = d_clean[feats].values; y = d_clean[star].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        mod = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1).fit(Xtr, ytr)
        yp = mod.predict(Xte)
        return {"type": "regression", "target": "rating (1-5)",
                "rmse": round(float(np.sqrt(mean_squared_error(yte, yp))), 2),
                "r2": round(float(r2_score(yte, yp)), 3),
                "train": len(Xtr), "test": len(Xte),
                "fi": {f: round(v, 4) for f, v in zip(feats, mod.feature_importances_)},
                "model": mod}
    except Exception: return None


def _churn_predictor(dfs):
    try:
        o = dfs.get("orders")
        u = dfs.get("users")
        if o is None or u is None or len(o) < 50: return None
        wu = _c(o, "worker_username")
        cu = _c(o, "client_username")
        st = _c(o, "status")
        cm = _c(o, "commission")
        if not all([wu, cu, st]): return None
        grp = o.groupby(cu).agg(
            total_orders=("id", "count"),
            total_commission=(cm, "sum") if cm else ("id", "count"),
            cancelled=("id", lambda x: sum(
                o.loc[x.index, st].str.upper() == "CANCELLED")),
        ).reset_index()
        grp.columns = ["username", "total_orders", "total_spent", "cancelled"]
        grp["cancel_rate"] = (grp["cancelled"] / grp["total_orders"] * 100).round(1)
        grp["has_more_than_one"] = (grp["total_orders"] > 2).astype(int)
        if grp["has_more_than_one"].nunique() < 2: return None
        gov = _c(u, "governorate")
        if gov:
            u_g = u[[cu if cu in u.columns else "username", gov]].copy()
            u_g.columns = ["username", "governorate"]
            grp = grp.merge(u_g, on="username", how="left")
            if "governorate" in grp.columns:
                le = LabelEncoder()
                grp["_gov_enc"] = le.fit_transform(grp["governorate"].fillna("Unknown"))
        feats = [c for c in ["total_spent", "cancel_rate", "_gov_enc"]
                 if c in grp.columns]
        if not feats: return None
        target = "has_more_than_one"
        cleaned = clean_for_ml(grp[feats + [target]], target_col=target)
        d_clean = cleaned["df"]
        if len(d_clean) < 30: return None
        X = d_clean[feats].values; y = d_clean[target].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        mod = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1).fit(Xtr, ytr)
        yp = mod.predict(Xte)
        cr = classification_report(yte, yp, output_dict=True, zero_division=0)
        return {"type": "classification", "target": "client re-order (>2 orders)",
                "accuracy": round(float(accuracy_score(yte, yp)), 3),
                "f1": round(float(cr["1"]["f1-score"]), 3),
                "train": len(Xtr), "test": len(Xte),
                "fi": {f: round(v, 4) for f, v in zip(feats, mod.feature_importances_)},
                "model": mod}
    except Exception: return None


def _worker_recommender(dfs):
    try:
        w = dfs.get("workers")
        if w is None or len(w) < 5: return None
        prof = _c(w, "profession")
        exp = _c(w, "experience_years")
        ar = _c(w, "average_rating")
        cj = _c(w, "completed_jobs")
        hr = _c(w, "hourly_rate")
        mc = _c(w, "minimum_charge")
        ac = _c(w, "accept_rate")
        un = _c(w, "username")
        nm = _c(w, "name")
        if not all([prof, un]): return None
        cols = [un, prof]
        if nm: cols.append(nm)
        if ar: cols.append(ar)
        if exp: cols.append(exp)
        if cj: cols.append(cj)
        if hr: cols.append(hr)
        if mc: cols.append(mc)
        if ac: cols.append(ac)
        d = w[cols].copy().dropna(subset=[prof, un])
        if ar: d[ar] = d[ar].fillna(d[ar].median())
        if exp: d[exp] = d[exp].fillna(d[exp].median())
        if cj: d[cj] = d[cj].fillna(d[cj].median())
        if hr: d[hr] = d[hr].fillna(d[hr].median())
        if mc: d[mc] = d[mc].fillna(d[mc].median())
        if ac: d[ac] = d[ac].fillna(d[ac].median())
        # Normalize numeric columns for fair scoring
        scaler = StandardScaler()
        numeric_scored = []
        if ar:
            d["_rating_sc"] = (d[ar] - d[ar].min()) / (d[ar].max() - d[ar].min() + 0.001)
            numeric_scored.append(("_rating_sc", 0.30))
        if ac:
            d["_accept_sc"] = d[ac] / 100.0
            numeric_scored.append(("_accept_sc", 0.20))
        if exp:
            d["_exp_sc"] = (d[exp] - d[exp].min()) / (d[exp].max() - d[exp].min() + 0.001)
            numeric_scored.append(("_exp_sc", 0.15))
        if cj:
            d["_jobs_sc"] = (d[cj] - d[cj].min()) / (d[cj].max() - d[cj].min() + 0.001)
            numeric_scored.append(("_jobs_sc", 0.10))
        if hr:
            # Invert: lower hourly rate is better for client
            d["_price_sc"] = 1 - (d[hr] - d[hr].min()) / (d[hr].max() - d[hr].min() + 0.001)
            numeric_scored.append(("_price_sc", 0.15))
        if mc:
            d["_min_sc"] = 1 - (d[mc] - d[mc].min()) / (d[mc].max() - d[mc].min() + 0.001)
            numeric_scored.append(("_min_sc", 0.10))
        if not numeric_scored: return None
        d["_score"] = sum(d[col] * wgt for col, wgt in numeric_scored)
        d["_score"] = (d["_score"] * 100).round(1)
        result = []
        for p in d[prof].unique():
            top = d[d[prof] == p].sort_values("_score", ascending=False).head(10)
            for _, row in top.iterrows():
                rec = {"profession": p, "score": row["_score"]}
                if un: rec["username"] = row[un]
                if nm: rec["name"] = row[nm]
                if ar: rec["rating"] = round(row[ar], 2)
                if exp: rec["experience_years"] = int(row[exp])
                if cj: rec["completed_jobs"] = int(row[cj])
                if hr: rec["hourly_rate"] = round(row[hr], 2)
                if ac: rec["accept_rate"] = round(row[ac], 1)
                result.append(rec)
        return {"type": "ranking", "target": "top workers per profession",
                "records": result[:50],
                "weights": {k: v for k, v in numeric_scored},
                "professions": d[prof].nunique(),
                "workers_scored": len(d)}
    except Exception: return None


def _worker_clustering(dfs):
    try:
        w = dfs.get("workers")
        if w is None or len(w) < 10: return None
        feats_spec = [
            ("average_rating", "avg_rating"),
            ("accept_rate", "accept_rate"),
            ("experience_years", "exp_years"),
            ("completed_jobs", "completed_jobs"),
            ("hourly_rate", "hourly_rate"),
        ]
        cols = []
        rename = {}
        for orig, alias in feats_spec:
            c = _c(w, orig)
            if c:
                cols.append(c)
                rename[c] = alias
        if len(cols) < 3: return None
        d = w[cols].copy().rename(columns=rename).dropna()
        if len(d) < 10: return None
        scaler = StandardScaler()
        X = scaler.fit_transform(d.values)
        n_clusters = min(4, len(d))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X)
        d["cluster"] = km.labels_
        labels = {0: "⭐ Elite", 1: "🔹 Average", 2: "📈 Growing", 3: "⚠️ Needs Improvement"}
        un = _c(w, "username")
        nm = _c(w, "name")
        prof = _c(w, "profession")
        extra = {}
        if un: extra["username"] = w.loc[d.index, un]
        if nm: extra["name"] = w.loc[d.index, nm]
        if prof: extra["profession"] = w.loc[d.index, prof]
        if extra:
            d = d.join(pd.DataFrame(extra, index=d.index))
        d["label"] = d["cluster"].map(labels).fillna("Cluster " + d["cluster"].astype(str))
        centroids = scaler.inverse_transform(km.cluster_centers_)
        cdf = pd.DataFrame(centroids, columns=d.columns[:len(cols)])
        cdf["cluster"] = range(n_clusters)
        cdf["label"] = cdf["cluster"].map(labels).fillna("Cluster " + cdf["cluster"].astype(str))
        dists = km.transform(X).min(axis=1)
        d["_dist"] = dists
        return {"type": "clustering", "target": "worker segments",
                "n_clusters": n_clusters,
                "workers": len(d),
                "assignments": d[["label"] + list(extra.keys())].to_dict("records")[:100],
                "centroids": cdf.to_dict("records"),
                "features": list(rename.values()),
                "model": km}
    except Exception: return None


def generate_recommendations(dfs, ml_results):
    recs = []

    # ── 1. 📍 Where Should We Expand? ──
    try:
        geo = geographic_opportunity(dfs)
        if geo:
            for g in sorted(geo, key=lambda x: x["gap"], reverse=True)[:5]:
                if g["gap"] <= 0:
                    continue
                recs.append({
                    "type": "expansion",
                    "title": f"📍 توسع في {g['governorate']}",
                    "detail": f"الفجوة: {g['gap']} طلب غير مخدوم. {g['workers']} عامل فقط مقابل {g['demand']} طلب. افتح فرع أو وفّر عمال.",
                    "priority": "high" if g["gap"] > 50 else "medium",
                })
    except Exception:
        pass

    # ── 2. 👷 Workforce Planning ──
    try:
        o = dfs.get("orders")
        w = dfs.get("workers")
        if o is not None and w is not None:
            prof_c = _c(w, "profession")
            wg = _c(w, "governorate")
            cat_c = _c(o, "category")
            if prof_c and cat_c:
                demand = o[cat_c].value_counts()
                supply = w[prof_c].value_counts()
                for svc in sorted(set(list(supply.index) + list(demand.index))):
                    d = demand.get(svc, 0)
                    s = supply.get(svc, 0)
                    if d > 0 and (s == 0 or d / s > 4):
                        needed = max(1, int(d / 3 - s))
                        gov_detail = ""
                        if wg and svc in w[prof_c].values:
                            w_gov = w[w[prof_c] == svc][wg].value_counts()
                            if len(w_gov):
                                gov_detail = f" في {w_gov.idxmax()}"
                        recs.append({
                            "type": "workforce",
                            "title": f"👷 احتياج: {svc}",
                            "detail": f"مطلوب ~{needed} عامل{gov_detail}. الطلب {d} مقابل {s} عامل فقط.",
                            "priority": "high" if d / s > 6 else "medium",
                        })
    except Exception:
        pass

    # ── 3. 💰 Revenue Opportunity ──
    try:
        o = dfs.get("orders")
        if o is not None and len(o) >= 5:
            cat_c = _c(o, "category")
            cm = _c(o, "commission")
            if cat_c and cm:
                svc_comm = o.groupby(cat_c)[cm].agg(["mean", "count"]).reset_index()
                svc_comm.columns = [cat_c, "avg_commission", "order_count"]
                oa = svc_comm["avg_commission"].mean()
                for _, row in svc_comm.iterrows():
                    svc = row[cat_c]
                    if row["avg_commission"] > oa * 1.2 and row["order_count"] >= 5:
                        recs.append({
                            "type": "revenue",
                            "title": f"💰 {svc} — فرصة ربح عالية",
                            "detail": f"متوسط العمولة {row['avg_commission']:.0f} EGP أعلى من المتوسط ({oa:.0f} EGP). ركّز على تسويقها.",
                            "priority": "medium",
                        })
                    elif row["avg_commission"] < oa * 0.7 and row["order_count"] >= 5:
                        recs.append({
                            "type": "revenue",
                            "title": f"💰 {svc} — يمكن تحسين الأسعار",
                            "detail": f"متوسط العمولة {row['avg_commission']:.0f} EGP أقل من المتوسط ({oa:.0f} EGP). راجع التسعير.",
                            "priority": "low",
                        })
    except Exception:
        pass

    # ── 4. 👥 Customer Retention ──
    try:
        ch = ml_results.get("churn_predictor")
        if ch:
            acc = ch.get("accuracy", 0)
            recs.append({
                "type": "retention",
                "title": "👥 عملاء معرضون للخسارة",
                "detail": f"نموذج الولاء بدقة {acc:.0%}. العملاء قليلو الطلب ومرتفعو الإلغاء الأكثر عرضة للترك. قدّم خصومات وعروض.",
                "priority": "high",
            })
    except Exception:
        pass

    # ── 5. ⚠️ Business Risk Detection ──
    try:
        o = dfs.get("orders")
        r = dfs.get("ratings")
        if o is not None and len(o):
            st = _c(o, "status")
            if st:
                cancel_rate = (o[st].str.upper() == "CANCELLED").sum() / len(o) * 100
                if cancel_rate > 20:
                    recs.append({
                        "type": "risk",
                        "title": "⚠️ ارتفاع معدل الإلغاء",
                        "detail": f"نسبة الإلغاء {cancel_rate:.0f}% — فوق الحد الآمن (20%). راجع الأسباب.",
                        "priority": "high",
                    })
        if r is not None and len(r):
            star = _c(r, "stars")
            if star and r[star].mean() < 3.5:
                recs.append({
                    "type": "risk",
                    "title": "⚠️ انخفاض تقييم العملاء",
                    "detail": f"متوسط التقييم {r[star].mean():.2f}/5 — أقل من المستهدف (4.0). حسّن الجودة.",
                    "priority": "high",
                })
        if o is not None:
            cat_c = _c(o, "category")
            st = _c(o, "status")
            if cat_c and st:
                per_cat = o.groupby(cat_c)[st].apply(
                    lambda x: (x.str.upper() == "CANCELLED").sum() / len(x) * 100
                ).reset_index()
                for _, row in per_cat[per_cat.iloc[:, 1] > 25].iterrows():
                    recs.append({
                        "type": "risk",
                        "title": f"⚠️ {row.iloc[0]}: إلغاء {row.iloc[1]:.0f}%",
                        "detail": "نسبة إلغاء مرتفعة. راجع جودة العمال في هذه الخدمة.",
                        "priority": "high",
                    })
    except Exception:
        pass

    # ── 7. 📈 Future Demand Forecast ──
    try:
        fc = demand_forecast(dfs)
        if fc:
            for f in [x for x in fc if x.get("direction") == "up"][:5]:
                recs.append({
                    "type": "future_demand",
                    "title": f"📈 طلب متزايد: {f['service']}",
                    "detail": f"متوقع {f['predicted']} طلب الشهر القادم (حالياً {f['current']}). وفّر عمال كافيين.",
                    "priority": "medium",
                })
            for f in [x for x in fc if x.get("direction") == "down"][:3]:
                recs.append({
                    "type": "future_demand",
                    "title": f"📉 طلب متناقص: {f['service']}",
                    "detail": f"متوقع {f['predicted']} طلب (حالياً {f['current']}). قلّل عدد العمال في هذه الخدمة.",
                    "priority": "low",
                })
    except Exception:
        pass

    # ── 8. 🧠 AI Business Advisor ──
    try:
        insights = []
        ch = ml_results.get("churn_predictor")
        if ch and ch.get("accuracy", 0) > 0.8:
            insights.append("نموذج ولاء العملاء دقيق ويمكن الاعتماد عليه")
        cc = ml_results.get("completion_classifier")
        if cc and cc.get("accuracy", 0) > 0.65:
            insights.append("نموذج إتمام الطلب جيد — استخدم أهم الميزات لتحسين الإتمام")
        wr = ml_results.get("worker_recommender")
        if wr:
            insights.append(f"تم تقييم {wr.get('workers_scored',0)} عامل عبر {wr.get('professions',0)} مهنة")
        models_ran = sum(1 for k in ["commission_predictor", "completion_classifier", "churn_predictor", "worker_recommender"] if k in ml_results)
        if models_ran >= 3:
            insights.append(f"✅ {models_ran}/4 نماذج رئيسية تعمل بكفاءة")
        else:
            insights.append(f"⚠️ {models_ran}/4 نماذج رئيسية فقط — زد حجم البيانات")
        if insights:
            recs.append({
                "type": "advisor",
                "title": "🧠 ملخص المستشار الذكي",
                "detail": " • ".join(insights),
                "priority": "medium",
            })
    except Exception:
        pass

    prio_map = {"high": 0, "medium": 1, "low": 2}
    recs.sort(key=lambda x: prio_map.get(x.get("priority", "low"), 2))
    return recs[:25]


def demand_forecast(dfs):
    try:
        o = dfs.get("orders")
        if o is None or len(o) < 10: return None
        cat_c = _c(o, "category")
        if not cat_c: return None
        o = o.copy()
        dc = _c(o, "created_at")
        if dc: o["_month"] = o[dc].dt.to_period("M").astype(str)
        else: return None
        forecast = []
        for svc in o[cat_c].unique():
            svc_o = o[o[cat_c] == svc]
            ts = svc_o.groupby("_month").size().reset_index(name="orders")
            ts = ts.sort_values("_month")
            if len(ts) < 2: continue
            ts["t"] = range(len(ts))
            X = ts[["t"]].values; y = ts["orders"].values
            lr = LinearRegression().fit(X, y)
            next_t = len(ts)
            pred = max(0, round(lr.predict([[next_t]])[0]))
            ts["_pred"] = lr.predict(ts[["t"]].values)
            ts["_pred"] = ts["_pred"].clip(0)
            current = y[-1]
            direction = "up" if pred > current else "down"
            forecast.append({
                "service": svc,
                "current": int(current),
                "predicted": pred,
                "direction": direction,
                "months": ts.to_dict("records"),
                "trend_slope": round(lr.coef_[0], 2),
            })
        if not forecast: return None
        forecast.sort(key=lambda x: x["predicted"], reverse=True)
        return forecast
    except Exception: return None


def optimal_pricing_zones(dfs):
    try:
        w = dfs.get("workers")
        if w is None or len(w) < 10: return None
        prof = _c(w, "profession")
        hr = _c(w, "hourly_rate")
        ar = _c(w, "average_rating")
        cj = _c(w, "completed_jobs")
        if not all([prof, hr, ar]): return None
        d = w[[prof, hr, ar]].dropna()
        if len(d) < 10: return None
        d.columns = ["profession", "hourly_rate", "avg_rating"]
        zones = []
        for p in d["profession"].unique():
            sub = d[d["profession"] == p]
            if len(sub) < 5: continue
            q = sub["hourly_rate"].quantile([0.25, 0.5, 0.75])
            rated_high = sub[sub["avg_rating"] >= sub["avg_rating"].quantile(0.66)]
            opt_price = rated_high["hourly_rate"].median() if len(rated_high) > 2 else q[0.5]
            zones.append({
                "profession": p,
                "workers": len(sub),
                "p25": round(q[0.25], 1), "p50": round(q[0.5], 1), "p75": round(q[0.75], 1),
                "opt_price": round(opt_price, 1),
                "avg_rating": round(sub["avg_rating"].mean(), 2),
                "top_avg_rating": round(rated_high["avg_rating"].mean(), 2) if len(rated_high) > 2 else 0,
            })
        return zones
    except Exception: return None


def geographic_opportunity(dfs):
    try:
        o = dfs.get("orders")
        w = dfs.get("workers")
        u = dfs.get("users")
        if o is None or u is None or w is None: return None
        og = _c(o, "client_username")
        wg = _c(w, "governorate")
        ug = _c(u, "governorate")
        uu = _c(u, "username")
        if not all([og, ug, wg, uu]): return None
        cl_gov = u[[uu, ug]].copy().dropna()
        cl_gov.columns = ["username", "governorate"]
        o2 = o.merge(cl_gov, left_on=og, right_on="username", how="left")
        if "governorate" not in o2.columns: return None
        demand = o2["governorate"].value_counts()
        supply = w[wg].value_counts()
        all_gov = sorted(set(list(demand.index) + list(supply.index)))
        result = []
        for g in all_gov:
            d = int(demand.get(g, 0))
            s = int(supply.get(g, 0))
            gap = d - s * 5
            result.append({
                "governorate": g,
                "demand": d,
                "workers": s,
                "gap": max(0, gap),
                "ratio": round(d / s, 1) if s > 0 else 999,
            })
        result.sort(key=lambda x: x["gap"], reverse=True)
        return result
    except Exception: return None
