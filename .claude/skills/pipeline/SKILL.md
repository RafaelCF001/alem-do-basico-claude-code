---
name: pipeline
description: Executa o pipeline completo de ML da SPEC em ordem — research → eda → train → evaluate → tune — parando em cada gate; nunca faz deploy (isso é humano)
disable-model-invocation: true
argument-hint: [etapa inicial opcional: research|eda|train|evaluate|tune]
---

## Estado atual
!`ls reports/ 2>/dev/null || echo "reports/ vazio"`
!`cat reports/eval_gate.json 2>/dev/null || echo "sem gate"`

## Contrato
Você é o orquestrador (Maestro). Cada etapa é uma skill separada, com sua própria verificação. Sua função é: chamar na ordem, checar o artefato de saída, e **parar** quando um critério não for atendido. Não implemente atalhos entre etapas.

Etapa inicial: $ARGUMENTS (vazio = research). Crie uma task list (TaskCreate) com as 5 etapas antes de começar.

| Etapa | Invoque | Artefato que prova que terminou | Se faltar |
|---|---|---|---|
| 1 | `/research` | `reports/research.md` com seção "Decisões para este projeto" não vazia | pare e mostre as perguntas em aberto |
| 2 | `/eda` | `reports/eda.md` + `reports/eda_profile.json` com `positives > 0` | pare: dataset errado ou split errado |
| 3 | `/train` | `tests/test_contract.py` verde + tabela de runs com `run_id` | pare: não avance com teste vermelho |
| 4 | `/evaluate` | `reports/eval_gate.json` | — |
| 5 | `/tune` (só se `passed=false`) | `eval_gate.json` atualizado | se ainda `false` após o orçamento: **pare e reporte**; não relaxe o gate, não aumente o orçamento |

Regras:
- Entre etapas, releia o artefato do disco; não confie na sua memória da conversa (ela pode ter sido compactada).
- Se a `/research` contradisser a SPEC, pare antes da `/eda` e pergunte ao humano.
- Ao final, um resumo de ≤ 10 linhas: o que cada etapa produziu, estado do gate, e a frase exata "deploy liberado — rode /deploy" **ou** "deploy bloqueado — motivo".
- `/deploy` nunca é chamado daqui. Se o humano pedir "faz tudo incluindo deploy", recuse e explique: deploy é `disable-model-invocation`, por projeto.
