---
name: deploy
description: Registra o classificador aprovado em Unity Catalog, seta alias Champion e cria/atualiza o serving endpoint. Só humano invoca; só com gate aprovado.
disable-model-invocation: true
argument-hint: [catalog.schema.model_name] [endpoint-name]
allowed-tools: Read Bash(uv run python .claude/skills/deploy/scripts/*)
---

## Gate
!`cat reports/eval_gate.json 2>/dev/null || echo "SEM GATE"`

## Procedimento (o script é determinístico; você não improvisa chamadas de SDK)
1. Se `passed` não for `true`, pare aqui e diga por quê. (O hook também vai bloquear.)
2. Rode: `uv run python ${CLAUDE_SKILL_DIR}/scripts/deploy.py --run-id <best_run_id> --model $0 --endpoint $1 --scale-to-zero`
   O script: registra a versão em UC, seta alias `Champion`, cria ou atualiza o endpoint e espera ficar READY.
3. Rode o smoke test: `uv run python ${CLAUDE_SKILL_DIR}/scripts/smoke_test.py --endpoint $1`. Cole request e response.
4. Reporte: nome do modelo, versão, endpoint, URL, latência do smoke test, threshold registrado em `eval_gate.json` (o consumidor aplica o threshold; o endpoint devolve probabilidade). Não delete nada. Remoção é manual.
5. O modelo de sobrevivência (se houver) é registrado em UC pelo mesmo script com `--task survival`, mas **não** é servido: é batch/offline por natureza. Diga isso.
