#!/bin/bash
# Guardian do Databricks. Recebe o tool call em JSON no stdin. exit 2 = bloqueia, stderr vira feedback pro Claude.
INPUT=$(cat)
CMD=$(echo "$INPUT" | jq -r '.tool_input.command // empty')
ROOT="${CLAUDE_PROJECT_DIR:-.}"
GATE="$ROOT/reports/eval_gate.json"

# 1. Nada de deletar endpoints/modelos via CLI, SDK ou REST sem um humano
if echo "$CMD" | grep -Eiq 'serving[-_]endpoints?(\.| )delete|delete_registered_model|delete_model_version|DELETE .*serving-endpoints'; then
  echo "Blocked: remoção de endpoint/modelo exige aprovação humana. Diga o que quer remover e por quê; eu executo manualmente." >&2
  exit 2
fi

# 2. Deploy só com gate aprovado
if echo "$CMD" | grep -Eq 'skills/deploy/scripts/deploy\.py'; then
  if [[ ! -f "$GATE" ]]; then
    echo "Blocked: reports/eval_gate.json não existe. Rode /evaluate antes de /deploy." >&2
    exit 2
  fi
  PASSED=$(jq -r '.passed // false' "$GATE" 2>/dev/null)
  if [[ "$PASSED" != "true" ]]; then
    echo "Blocked: eval_gate.json diz passed=$PASSED. Não faça deploy de modelo reprovado; melhore via /tune ou revise a SPEC com o humano." >&2
    exit 2
  fi
fi

# 3. Segredos não passam por linha de comando
if echo "$CMD" | grep -Eq 'DATABRICKS_TOKEN=|dapi[0-9a-f]{20,}|cat .*databrickscfg'; then
  echo "Blocked: token do Databricks em linha de comando ou leitura do databrickscfg. Use DATABRICKS_CONFIG_PROFILE." >&2
  exit 2
fi
exit 0
