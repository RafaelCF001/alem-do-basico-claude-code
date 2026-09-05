#!/bin/bash
# Guardian: bloqueia edição de arquivos sensíveis. exit 2 = bloqueia e manda stderr pro Claude.
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
FILE_PATH="${FILE_PATH//\\//}"
PROTECTED=(".env" "secrets/" ".git/" "uv.lock" "poetry.lock" "package-lock.json")
for p in "${PROTECTED[@]}"; do
  if [[ "$FILE_PATH" == *"$p"* ]]; then
    echo "Blocked: $FILE_PATH casa com padrão protegido '$p'. Peça ao humano." >&2
    exit 2
  fi
done
exit 0
