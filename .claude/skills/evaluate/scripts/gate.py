"""Gate determinístico: melhor run por métrica primária, decide passed/failed. Nunca editado pelo Claude."""
from __future__ import annotations
import argparse, json, pathlib, sys
import mlflow

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--task", default="classification")
    ap.add_argument("--metric", action="append", required=True, help="repetível; o primeiro é o primário (maior = melhor)")
    ap.add_argument("--min", action="append", type=float, required=True)
    ap.add_argument("--out", default="reports/eval_gate.json")
    ap.add_argument("--set-threshold", type=float, default=None)
    ap.add_argument("--expected-cost", type=float, default=None)
    a = ap.parse_args()
    if len(a.metric) != len(a.min):
        print("--metric e --min devem vir em pares", file=sys.stderr); return 1

    mlflow.set_tracking_uri("databricks")
    exp = mlflow.get_experiment_by_name(a.experiment)
    if exp is None:
        print(f"experimento {a.experiment} não existe", file=sys.stderr); return 1
    primary = a.metric[0]
    runs = mlflow.search_runs([exp.experiment_id], filter_string=f"tags.task = '{a.task}' and metrics.{primary} > 0",
                              order_by=[f"metrics.{primary} DESC"], max_results=100)
    if runs.empty:
        print(f"nenhum run com task={a.task} e métrica {primary}", file=sys.stderr); return 1
    best = runs.iloc[0]
    checks = {}
    for m, mn in zip(a.metric, a.min):
        v = best.get(f"metrics.{m}")
        checks[m] = {"value": None if v is None else float(v), "min": mn, "ok": bool(v is not None and float(v) >= mn)}
    passed = all(c["ok"] for c in checks.values())
    result = {"task": a.task, "passed": passed, "best_run_id": best["run_id"],
              "model": best.get("tags.model", best.get("params.model", "?")),
              "checks": checks, "runs_considered": int(len(runs))}
    if a.set_threshold is not None: result["threshold"] = a.set_threshold
    if a.expected_cost is not None: result["expected_cost"] = a.expected_cost
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))
    return 0 if passed else 3

if __name__ == "__main__":
    raise SystemExit(main())
