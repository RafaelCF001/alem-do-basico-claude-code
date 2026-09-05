"""Deploy determinístico: run aprovado -> Unity Catalog -> alias Champion -> serving endpoint."""
from __future__ import annotations
import argparse, sys, time
import mlflow
from mlflow import MlflowClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", required=True, help="catalog.schema.name")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--workload-size", default="Small")
    ap.add_argument("--scale-to-zero", action="store_true")
    a = ap.parse_args()

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mv = mlflow.register_model(f"runs:/{a.run_id}/model", a.model)
    MlflowClient().set_registered_model_alias(a.model, "Champion", mv.version)
    print(f"registrado {a.model} v{mv.version} alias=Champion")

    w = WorkspaceClient()
    cfg = EndpointCoreConfigInput(served_entities=[ServedEntityInput(
        entity_name=a.model, entity_version=str(mv.version),
        workload_size=a.workload_size, scale_to_zero_enabled=a.scale_to_zero)])
    names = {e.name for e in w.serving_endpoints.list()}
    if a.endpoint in names:
        w.serving_endpoints.update_config(name=a.endpoint, config=cfg)
        print(f"endpoint {a.endpoint} atualizado para v{mv.version}")
    else:
        w.serving_endpoints.create(name=a.endpoint, config=cfg)
        print(f"endpoint {a.endpoint} criado")

    for _ in range(60):
        st = w.serving_endpoints.get(a.endpoint).state
        ready = getattr(st, "ready", None)
        print(f"state: ready={ready} config_update={getattr(st, 'config_update', None)}")
        if str(ready).endswith("READY"):
            print(f"URL: {w.config.host}/serving-endpoints/{a.endpoint}/invocations")
            return 0
        time.sleep(20)
    print("timeout esperando READY", file=sys.stderr)
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
