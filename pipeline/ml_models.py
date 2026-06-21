import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
