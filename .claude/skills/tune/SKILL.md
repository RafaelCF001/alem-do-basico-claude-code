---
name: tune
description: Busca de hiperparâmetros com orçamento fixo da SPEC para o melhor candidato reprovado no gate; CV temporal; loga no MLflow e re-roda o gate
argument-hint: [n_iter opcional, nunca acima do orçamento]
allowed-tools: Bash(uv run python *) Bash(uv run pytest *) Read
---

## Estado do gate
!`cat reports/eval_gate.json 2>/dev/null || echo "sem gate — rode /evaluate"`

## Orçamento e técnicas
!`sed -n '/## Orçamento/,$p' SPEC.md 2>/dev/null`
!`grep -i -A2 "tuning\|hiperpar" reports/research.md 2>/dev/null | head -12`

## Procedimento
1. Parta do `model` em `eval_gate.json`. `src/tune.py` com `RandomizedSearchCV` e `TimeSeriesSplit(n_splits=3)` (nunca KFold aleatório — drift), `scoring="average_precision"`, `n_iter` = $ARGUMENTS ou o máximo da SPEC — nunca acima.
2. Espaço de busca pequeno; justifique cada hiperparâmetro em 1 linha. Inclua `class_weight`/`scale_pos_weight` no espaço.
3. Avaliação final SÓ no `data/test.parquet` salvo. Logue o melhor como run com tags `stage=tune`, `task=classification`, `model=<nome>`, `input_example`, `signature`.
4. Re-rode o gate (mesmos argumentos do /evaluate). Cole o JSON.
5. Ainda reprovado após o orçamento: PARE. Não aumente n_iter, não relaxe o gate. Reporte as 2 melhores tentativas e o que a pesquisa sugere como próximo passo (features, não hiperparâmetros).
