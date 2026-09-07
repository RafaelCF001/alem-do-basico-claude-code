# SPEC — detecção de fraude em transações (PaySim)
<!-- Gerado por entrevista. Preencha antes de qualquer código. /clear depois. A /research pode contestar; contestação vira "pergunta em aberto", não edição silenciosa. -->

## Objetivo
Uma frase: que decisão o modelo apoia (ex.: bloquear/revisar transação em tempo real) e quem consome a probabilidade.

## Dataset
- Fonte: PaySim (`kagglehub.dataset_download("ealaxi/paysim1")`), ~6,3M transações, ~0,13% fraude, `step` = hora (30 dias)
- Alvo: `isFraud`. Feature proibida: `isFlaggedFraud` (regra pós-hoc)
- Split temporal: treino `step ≤ 600`, teste `step > 600`
- Amostra para a demo: treino downsampled em negativos (ex.: 300k) mantendo TODOS os positivos; teste intacto — registrar isso no MLflow

## Tarefas
1. **Classificação por transação** (servida): P(fraude | transação)
2. **Sobrevivência por conta de destino** (offline): tempo até a conta receber a primeira fraude; Kaplan-Meier + Cox. Uso: priorizar monitoramento de contas

## Critério de sucesso (gate)
- Classificação: `pr_auc` ≥ **0.80** no teste; `recall_at_p90` ≥ **0.60**; `brier` ≤ 0.002
- Sobrevivência: `c_index` ≥ **0.70** no teste
- Threshold escolhido por custo: falso negativo = valor da transação; falso positivo = R$ 15 (revisão manual) — ajuste na entrevista
- Latência do endpoint: p95 < 300 ms para 1 registro
- Baseline obrigatório: regressão logística com `class_weight="balanced"`

## Fora de escopo (não faça)
- Grafos de contas / GNN
- Deep learning
- Reamostragem fora do pipeline; qualquer feature com informação posterior ao `step`
- Deploy sem `reports/eval_gate.json` com `"passed": true`
- Servir o modelo de sobrevivência (é batch)
- Deletar endpoints sem aprovação humana

## Verificação end-to-end
`uv run python .claude/skills/deploy/scripts/smoke_test.py --endpoint <nome>` devolve probabilidade em [0,1] para uma transação do teste, com latência dentro do gate.

## Orçamento
- Tuning: ≤ 25 iterações de RandomizedSearchCV, CV temporal 3 folds
- Modelos: haiku no /eda; sonnet em /research, /train, /evaluate, /tune; opus só no @ml-reviewer
- Tempo de demo: 25 min; se treino > 3 min, reduzir amostra de treino
