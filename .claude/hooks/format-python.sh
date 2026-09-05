#!/bin/bash
# Determinístico: formata todo .py editado. Saída em exit 0 vai pro debug log, não pro contexto.
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
if [[ "$FILE_PATH" == *.py ]]; then
  uv run ruff format "$FILE_PATH" >/dev/null 2>&1 || true
  uv run ruff check --fix "$FILE_PATH" >/dev/null 2>&1 || true
fi
exit 0
