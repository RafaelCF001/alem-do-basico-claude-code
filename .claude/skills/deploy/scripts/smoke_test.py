"""Smoke test do endpoint com o input_example registrado. Evidência, não afirmação."""
from __future__ import annotations
import argparse, json, time
import pandas as pd
from databricks.sdk import WorkspaceClient

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    test = pd.read_parquet("data/test.parquet")
    drop = [c for c in ["isFraud", "isFlaggedFraud", "nameOrig", "nameDest"] if c in test.columns]
    row = test.drop(columns=drop).iloc[[0]]
    w = WorkspaceClient()
    lat = []
    for _ in range(a.n):
        t = time.perf_counter()
        r = w.serving_endpoints.query(name=a.endpoint, dataframe_records=row.to_dict("records"))
        lat.append((time.perf_counter() - t) * 1000)
    print("request:", json.dumps(row.to_dict("records")))
    print("response:", r.predictions)
    print(f"latency ms: {[round(x) for x in lat]}  p95≈{sorted(lat)[int(0.95*(a.n-1))]:.0f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
