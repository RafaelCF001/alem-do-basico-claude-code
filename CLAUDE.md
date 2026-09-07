# fraud-serving — prevenção a fraudes (classificação + sobrevivência) servida no Databricks

<!-- Raiz: só o que vale em qualquer pasta. Meta < 200 linhas. -->

## Fonte da verdade
- `SPEC.md` define objetivo, métricas-alvo, dataset, custos e o que está FORA de escopo. Conflito com a spec → pare e pergunte.
- `reports/research.md` é a base técnica: decisões de modelagem citam a seção "Decisões para este projeto". Sem fonte lá, é hipótese, não decisão.
- `reports/eval_gate.json` é o único sinal aceito de "modelo bom o suficiente". Leia o arquivo; não afirme.

## Comandos
- Deps: `uv sync`
- Testes: `uv run pytest tests/ -q` (um arquivo por vez)
- Lint/format: `uv run ruff check . && uv run ruff format .`
- Pipeline: `/pipeline` (= `/research` → `/eda` → `/train` → `/evaluate` → `/tune`); `/deploy` só por humano

## Dados e split (não negociável)
- Dataset PaySim (`kagglehub`, `ealaxi/paysim1`); `step` = hora simulada (1–744). Fallback sem Kaggle: OpenML `creditcard` (só classificação).
- Split **temporal** feito UMA vez em `src/data.py`: treino = `step ≤ 600`, teste = `step > 600`. Nunca re-splitar; nunca `train_test_split` aleatório.
- Fraude só existe em `type ∈ {TRANSFER, CASH_OUT}`; decisão de filtrar ou não vem da research + EDA, registrada na SPEC.
- Features proibidas (`src/features/FORBIDDEN_FEATURES`): `isFlaggedFraud`, ids crus `nameOrig`/`nameDest`, qualquer coluna calculada com dados posteriores ao `step` da transação.
- Toda métrica reportada vem de `data/test.parquet`. PR-AUC é a primária; ROC-AUC é informativo.
- Sobrevivência: unidade = conta de destino; `duration` = steps desde a primeira transação recebida até a primeira fraude recebida; `event=1` se houve fraude no treino, senão censurado no `step` 600. Covariáveis só de antes do evento.

## Convenções que diferem do default
- Python 3.12, type hints em funções públicas, `logging` em vez de `print`
- Reamostragem (se usada) só via `imblearn.pipeline.Pipeline`, dentro do pipeline, nunca antes do split
- MLflow: `mlflow.set_tracking_uri("databricks")`, `mlflow.set_registry_uri("databricks-uc")`, experimento `/Shared/fraud-serving`; tags obrigatórias `stage`, `task`, `model`
- Credenciais só via `DATABRICKS_CONFIG_PROFILE` e `KAGGLE_*` no ambiente; nunca leia nem cite `~/.databrickscfg` ou `~/.kaggle/kaggle.json`

## Workflow
- IMPORTANT: mostre a saída do comando de teste/avaliação em vez de afirmar que passou
- Ao compactar, preserve: arquivos modificados, run_ids citados, estado de `reports/eval_gate.json`, e as "Decisões" da research
- Commits em inglês, imperativo; branch `feat/<tema>`
