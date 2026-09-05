#!/bin/bash
# Stop hook: não deixa o turno encerrar com testes quebrados quando houve mudança em src/.
# Claude Code desiste após 8 bloqueios seguidos; stop_hook_active evita loop.
INPUT=$(cat)
ACTIVE=$(echo "$INPUT" | jq -r '.stop_hook_active // false')
[[ "$ACTIVE" == "true" ]] && exit 0
if git diff --quiet HEAD -- src/ 2>/dev/null && [ -z "$(git ls-files --others --exclude-standard src/ 2>/dev/null)" ]; then
  exit 0   # nada mudou em src/, libera
fi
ls tests/test_*.py >/dev/null 2>&1 || exit 0   # sem testes ainda (início da demo), libera
if ! uv run pytest -q -x >/tmp/claude-tests.log 2>&1; then
  echo "Testes falhando após mudanças em src/. Corrija antes de encerrar. Últimas linhas:" >&2
  tail -20 /tmp/claude-tests.log >&2
  exit 2
fi
exit 0
