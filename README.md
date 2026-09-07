# fraud-serving

Kit de demonstração do Claude Code: prevenção a fraudes em transações financeiras (dataset **PaySim**), do zero a um
modelo servido no **Databricks**, guiado por spec, research, skills e testes de contrato — não por código escrito à mão.

O harness (`.claude/`, `CLAUDE.md`, `SPEC.template.md`, `pyproject.toml`) é o ponto de partida do kit. O restante
(`SPEC.md`, `src/`, `tests/`, `reports/`) é gerado ao vivo pelo Claude, dirigido pelas skills do pipeline — esse é o
propósito da demo. Veja o roteiro completo em [`DEMO.md`](DEMO.md).

## O que este projeto faz

- **Classificação por transação** (servida em tempo real): `P(fraude | transação)`, usada para auto-aprovar, bloquear
  ou mandar para revisão manual uma transferência/saque no momento em que ocorre.
- **Sobrevivência por conta de destino** (offline, batch): tempo até a conta receber a primeira fraude
  (Kaplan-Meier + Cox), usado para priorizar monitoramento de contas por risco.

Objetivo, dataset, métricas-alvo, custos e o que está fora de escopo estão em [`SPEC.md`](SPEC.md) — fonte da verdade
do projeto.

## Status atual

| Tarefa | Modelo | Métrica | Alvo | Resultado | Gate |
|---|---|---|---|---|---|
| Classificação | `hgb_calibrated` | PR-AUC | ≥ 0.80 | 0.989 | ✅ passou |
| Classificação | `hgb_calibrated` | recall@p90 | ≥ 0.60 | 0.982 | ✅ passou |
| Sobrevivência | `coxph` | c-index | ≥ 0.70 | 0.658 | ❌ não passou |

`reports/eval_gate.json` e `reports/eval_gate_survival.json` são o único sinal aceito de "modelo bom o suficiente"
(gerados pela skill `/evaluate`); apenas o classificador é servido — a sobrevivência roda offline e ainda não atinge
o alvo de c-index da SPEC.

## Pipeline

```
/research → /eda → /train → /evaluate → /tune
```

Orquestrado pela skill `/pipeline`; `/deploy` é sempre acionado por um humano. Cada etapa lê o que a anterior deixou
em `reports/` — nenhuma decisão de modelagem entra sem estar registrada em `reports/research.md` (seção "Decisões
para este projeto").

| Skill | O que faz |
|---|---|
| `research` | Pesquisa web sobre as técnicas do problema da SPEC; escreve `reports/research.md` |
| `eda` | EDA orientada a fraude (desbalanceamento, drift temporal, censura); números vêm de script determinístico |
| `train` | Testes de contrato → treina baseline e candidatos; loga tudo no MLflow do Databricks |
| `evaluate` | Compara runs contra o gate da SPEC, escolhe threshold por custo, grava `reports/eval_gate*.json` |
| `tune` | Busca de hiperparâmetros com orçamento fixo para o melhor candidato reprovado no gate |
| `deploy` | Registro no Unity Catalog + Model Serving; só roda com aprovação humana |

## Dados

- PaySim (`kagglehub`, `ealaxi/paysim1`), `step` = hora simulada (1–744). Fallback sem Kaggle: OpenML `creditcard`
  (só classificação).
- Split **temporal**, feito uma única vez em `src/data.py`: treino = `step ≤ 600`, teste = `step > 600`. Nunca split
  aleatório.
- Toda métrica reportada vem de `data/test.parquet`.

Detalhes de features proibidas, unidade da sobrevivência e convenções de modelagem estão em `CLAUDE.md`.

## Rodando localmente

```bash
uv sync
uv run pytest tests/ -q
uv run ruff check . && uv run ruff format .
```

Credenciais (Databricks, Kaggle) só via variáveis de ambiente — nunca em arquivo de projeto. Veja os pré-requisitos
completos em [`DEMO.md`](DEMO.md).

## Estrutura

```
.claude/           # skills, hooks, permissões — o harness do kit
CLAUDE.md          # convenções do projeto (fonte de verdade para o Claude)
SPEC.md            # objetivo, métricas-alvo, dataset, escopo
reports/           # research.md, eda.md, eval_gate*.json
src/               # data, features, train, calibrate, threshold, tune
tests/             # testes de contrato (split, features proibidas, reamostragem)
```
