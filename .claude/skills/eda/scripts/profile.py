"""Perfil determinístico para EDA de fraude. Escreve JSON + figuras. O Claude interpreta; não recalcula."""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np, pandas as pd

def q(s: pd.Series) -> dict:
    return {k: float(v) for k, v in s.quantile([.01, .1, .25, .5, .75, .9, .99]).round(4).items()}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--target", default="isFraud")
    ap.add_argument("--time-col", default="step"); ap.add_argument("--out", default="reports/eda_profile.json")
    ap.add_argument("--figures", default="reports/eda_figures"); ap.add_argument("--survival", default="data/survival_train.parquet")
    a = ap.parse_args()
    df = pd.read_parquet(a.train); y = df[a.target].astype(int)
    out: dict = {"shape": list(df.shape), "dtypes": df.dtypes.astype(str).to_dict(),
                 "nulls": df.isna().sum().to_dict(), "duplicates": int(df.duplicated().sum()),
                 "positives": int(y.sum()), "fraud_rate": float(y.mean())}
    if "type" in df: out["fraud_rate_by_type"] = df.groupby("type")[a.target].agg(["mean", "sum", "count"]).round(6).to_dict("index")
    if a.time_col in df:
        bins = pd.cut(df[a.time_col], bins=10)
        g = df.groupby(bins, observed=True)[a.target].agg(["mean", "sum", "count"])
        out["fraud_rate_by_time_window"] = {str(k): {kk: float(vv) for kk, vv in v.items()} for k, v in g.round(6).to_dict("index").items()}
    if "amount" in df:
        out["amount_quantiles"] = {"fraud": q(df.loc[y == 1, "amount"]), "legit": q(df.loc[y == 0, "amount"])}
    for c in ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        if c in df: out.setdefault("zero_balance_rate", {})[c] = {"fraud": float((df.loc[y == 1, c] == 0).mean()), "legit": float((df.loc[y == 0, c] == 0).mean())}
    if "nameDest" in df:
        fr = df[y == 1]["nameDest"].value_counts()
        out["dest_accounts"] = {"unique_total": int(df["nameDest"].nunique()), "with_fraud": int(fr.size),
                                "with_more_than_one_fraud": int((fr > 1).sum()), "max_frauds_one_account": int(fr.max()) if fr.size else 0}
    num = df.select_dtypes("number").drop(columns=[a.target], errors="ignore")
    corr = num.corrwith(y).abs().sort_values(ascending=False)
    out["abs_corr_with_target_top10"] = corr.head(10).round(4).to_dict()
    out["leakage_suspects"] = [c for c in corr.index if corr[c] > 0.95] + [c for c in ["isFlaggedFraud"] if c in df]
    out["constant_columns"] = [c for c in df.columns if df[c].nunique() <= 1]
    if "amount" in df: out["impossible_values"] = {"amount_le_0": int((df["amount"] <= 0).sum())}
    sp = pathlib.Path(a.survival)
    if sp.exists():
        s = pd.read_parquet(sp); ev = s["event"].astype(int)
        out["survival"] = {"n": int(len(s)), "events": int(ev.sum()), "censoring_rate": float(1 - ev.mean())}
        try:
            from lifelines import KaplanMeierFitter
            km = KaplanMeierFitter().fit(s["duration"], ev)
            out["survival"]["km_median"] = None if np.isinf(km.median_survival_time_) else float(km.median_survival_time_)
            out["survival"]["km_survival_at"] = {str(t): float(km.survival_function_at_times(t).iloc[0]) for t in [24, 168, 336, 720] if t <= s["duration"].max()}
        except ImportError:
            out["survival"]["km"] = "lifelines não instalado"
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fdir = pathlib.Path(a.figures); fdir.mkdir(parents=True, exist_ok=True)
        if a.time_col in df:
            ax = df.groupby(a.time_col)[a.target].mean().plot(figsize=(8, 3), title="fraud rate by step"); ax.figure.savefig(fdir / "fraud_rate_by_step.png", bbox_inches="tight"); plt.close("all")
        if "amount" in df:
            ax = np.log1p(df["amount"]).groupby(y).plot(kind="hist", bins=60, alpha=.5, legend=True, figsize=(8, 3), title="log1p(amount) by class"); plt.savefig(fdir / "amount_by_class.png", bbox_inches="tight"); plt.close("all")
        out["figures"] = sorted(str(p) for p in fdir.glob("*.png"))
    except ImportError:
        out["figures"] = "matplotlib não instalado"
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: out[k] for k in ["shape", "positives", "fraud_rate", "leakage_suspects", "dest_accounts", "survival"] if k in out}, indent=2, default=str))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
