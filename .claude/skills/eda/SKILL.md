---
name: eda
description: Análise exploratória orientada a fraude — desbalanceamento, drift temporal, agregações por conta, censura para sobrevivência, leakage; números vêm de script determinístico, o Claude interpreta
argument-hint: "foco opcional, ex. survival ou por tipo de transação"
model: haiku
allowed-tools: Read Bash(uv run python *) Edit(./reports/**)
---

## Base de conhecimento
!`sed -n '/## TL;DR/,/## Técnicas/p' reports/research.md 2>/dev/null || echo "sem research.md — rode /research primeiro; a EDA fica mais rasa sem ela"`
!`sed -n '/## Armadilhas de leakage/,/## Decisões/p' reports/research.md 2>/dev/null`

## SPEC
!`sed -n '/## Dataset/,/## Fora de escopo/p' SPEC.md 2>/dev/null`

## Estado dos dados
!`ls -la data/ 2>/dev/null || echo "data/ vazio"`

## Regras
- Analise SÓ `data/train.parquet`. Abrir `data/test.parquet` aqui é leakage de decisão; não faça.
- Os números vêm de `${CLAUDE_SKILL_DIR}/scripts/profile.py`. Você não recalcula na mão; você lê `reports/eda_profile.json` e interpreta.
- Foco: $ARGUMENTS (se vazio, cobertura completa).

## Procedimento
1. Se `data/train.parquet` não existir: rode `uv run python src/data.py`. Se `src/data.py` não existir, crie-o conforme CLAUDE.md e SPEC (download, split **temporal** por `step`, salva `data/train.parquet` e `data/test.parquet`, e a tabela de sobrevivência `data/survival_train.parquet` se a SPEC pedir).
2. Rode `uv run python ${CLAUDE_SKILL_DIR}/scripts/profile.py --train data/train.parquet --target isFraud --time-col step --out reports/eda_profile.json --figures reports/eda_figures`.
3. Escreva `reports/eda.md` com as seções, **cada uma com o número que a sustenta**:
   - Shape, tipos, nulos, duplicatas
   - **Desbalanceamento**: taxa de fraude global e por `type`; quantos positivos há de fato (isso dita CV e métricas)
   - **Drift temporal**: taxa de fraude por janela de `step`; volume por janela; existe período sem fraude?
   - **Alvo × features**: distribuição de `amount` por classe (quantis, não média), razões de saldo, transações que zeram saldo
   - **Por conta**: reincidência de `nameDest` em fraudes (contas mula), contas com > 1 fraude; isso justifica ou não o framing de sobrevivência
   - **Sobrevivência** (se a SPEC pedir): nº de contas, eventos, taxa de censura, mediana de tempo até evento pela KM (do profile), covariáveis candidatas disponíveis **antes** do evento
   - **Leakage**: `isFlaggedFraud` e qualquer coluna com correlação > 0,95 ou definida após o evento; liste o que **não** pode ser feature
   - **Qualidade**: valores impossíveis (saldo negativo, amount 0), colunas constantes
   - **Hipóteses de features** (≤ 5), cada uma ligada a uma evidência acima e, quando houver, à técnica em research.md
4. Termine com "Implicações para /train" em ≤ 6 linhas: protocolo de split, métrica primária, tratamento de desbalanceamento a testar, features proibidas.
5. Não treine nada. Não altere a SPEC.
