---
name: evaluate
description: Compara runs do MLflow contra o gate da SPEC (PR-AUC e, se houver, c-index), escolhe threshold por custo e grava reports/eval_gate.json — único sinal aceito de aprovação
allowed-tools: Bash(uv run python *) Read
---

## Gate da SPEC
!`sed -n '/## Critério de sucesso/,/## Fora de escopo/p' SPEC.md 2>/dev/null || echo "SPEC sem gate"`

## Protocolo recomendado pela pesquisa
!`sed -n '/## Métricas e protocolo/,/## Armadilhas/p' reports/research.md 2>/dev/null`

## Procedimento
1. Rode o gate determinístico com os valores da SPEC, por exemplo:
   `uv run python ${CLAUDE_SKILL_DIR}/scripts/gate.py --experiment <exp> --task classification --metric pr_auc --min <valor> --metric recall_at_p90 --min <valor>`
   e, se houver sobrevivência:
   `uv run python ${CLAUDE_SKILL_DIR}/scripts/gate.py --experiment <exp> --task survival --metric c_index --min <valor> --out reports/eval_gate_survival.json`
2. Cole o(s) JSON(s).
3. Threshold: se `passed=true`, rode `uv run python src/threshold.py --run-id <best>` (crie se não existir: varre thresholds no teste salvo e escolhe o que minimiza custo esperado com os custos da SPEC). Registre `threshold` e custo no JSON via `--set-threshold`.
4. `passed=false`: diga em 3 linhas a distância até o gate e recomende /tune com o orçamento da SPEC. **Não** ajuste o threshold do gate, **não** troque a métrica.
5. `passed=true`: liste o run vencedor e diga "deploy liberado". Não faça deploy aqui.

O script decide; você interpreta.
