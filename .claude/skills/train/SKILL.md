---
name: train
description: Treina baseline e candidatos de classificação de fraude (e modelo de sobrevivência se a SPEC pedir) com testes de contrato antes; loga tudo no MLflow do Databricks
argument-hint: [modelos opcionais, ex. "logreg hgb" ou "cox rsf"]
allowed-tools: Bash(uv run pytest *) Bash(uv run python *) Bash(uv run ruff *) Read
---

## Base de conhecimento
!`sed -n '/## Decisões para este projeto/,/## Perguntas em aberto/p' reports/research.md 2>/dev/null || echo "sem research.md"`
!`sed -n '/Implicações para \/train/,$p' reports/eda.md 2>/dev/null || echo "sem eda.md — rode /eda"`

## Estado
!`ls src/ tests/ 2>/dev/null`

## TDD de contrato (antes de qualquer treino)
Garanta `tests/test_contract.py` cobrindo:
- `load_split()` devolve treino/teste **temporais**: `max(train.step) < min(test.step)`, sem interseção de índices
- Nenhuma coluna de `FORBIDDEN_FEATURES` (de `src/features/__init__.py`, inclui `isFlaggedFraud`, `nameOrig`, `nameDest` cru) entra em `X`
- `build_pipeline(name)` devolve `sklearn.Pipeline`; qualquer scaler/encoder/reamostragem está **dentro** dele
- `pipeline.predict_proba(input_example)` funciona para 1 linha com o schema de `data/train.parquet`
- Se a SPEC pede sobrevivência: `build_survival_table()` tem colunas `duration`, `event` e só covariáveis anteriores ao evento
Rode; MOSTRE o vermelho; implemente; MOSTRE o verde. Não edite os testes para passarem.

## Treino
1. `src/train.py`: baseline `LogisticRegression(class_weight="balanced")` + candidatos ($ARGUMENTS ou `hgb` = `HistGradientBoostingClassifier`). Trate desbalanceamento conforme research.md (pesos de classe primeiro; reamostragem só se a pesquisa sustentar, e dentro do pipeline via imblearn).
2. Métricas no `data/test.parquet` salvo: `pr_auc` (average precision), `roc_auc`, `recall_at_p90` (recall com precisão ≥ 0,90), `brier`. Escolha de threshold fica para /evaluate.
3. Se a SPEC pede sobrevivência: `src/train_survival.py` com Kaplan-Meier (descritivo) e `CoxPHFitter` (lifelines); métrica `c_index` no teste; logar como run separado com tag `task=survival`.
4. Cada run em MLflow (`mlflow.set_tracking_uri("databricks")`, experimento da SPEC): params, métricas, `input_example`, `signature`, tags `stage=train`, `task=classification|survival`, `model=<nome>`.
5. Rode e cole a tabela final (modelo, task, run_id, métricas). Não declare vencedor.

Evidência exigida: pytest verde e tabela de runs com run_id.
