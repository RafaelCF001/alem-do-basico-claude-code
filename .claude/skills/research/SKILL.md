---
name: research
description: Pesquisa web e revisão da literatura sobre as técnicas do problema da SPEC (fraude, desbalanceamento, sobrevivência, métricas); escreve reports/research.md, que as outras skills usam como base
argument-hint: [tema opcional para aprofundar, ex. "calibração" ou "survival para fraude"]
context: fork
agent: general-purpose
background: false
model: sonnet
allowed-tools: WebSearch WebFetch Read Edit(./reports/**) Bash(ls *) Bash(date *)
---

## Problema (da SPEC)
!`sed -n '1,60p' SPEC.md 2>/dev/null || echo "SPEC.md não existe — peça a entrevista antes"`

## Já pesquisado
!`ls reports/research*.md 2>/dev/null && head -20 reports/research.md || echo "nenhuma pesquisa anterior"`

## Objetivo
Produzir `reports/research.md`: uma revisão curta, com fontes, que responde **o que a literatura recomenda para ESTE problema** — não um survey genérico. Foco: $ARGUMENTS (se vazio: cobertura completa abaixo).

## Procedimento
1. Para cada tópico, faça 2–4 buscas (`WebSearch`) e leia 1–3 fontes (`WebFetch`). Prefira, nesta ordem: papers (arXiv, ACM, IEEE, JMLR), documentação oficial das bibliotecas (scikit-learn, lifelines, scikit-survival, imbalanced-learn, MLflow, Databricks), benchmarks reprodutíveis. Blog só se citar paper ou código.
2. Tópicos obrigatórios:
   - **Framing**: classificação por transação vs. detecção por conta vs. **análise de sobrevivência** (tempo até o primeiro evento de fraude / tempo até comprometimento de conta). Quando cada um faz sentido; o que a SPEC pede.
   - **Desbalanceamento**: reamostragem (SMOTE e variantes), pesos de classe, threshold moving, focal loss; evidência de quando ajuda vs. atrapalha (calibração!).
   - **Modelos**: baseline logístico; gradient boosting (LightGBM/XGBoost/HistGB) como estado da prática tabular; isolation forest / autoencoder como não-supervisionado; para sobrevivência: Kaplan-Meier, Cox PH, Random Survival Forest, gradient boosting survival.
   - **Features**: agregações temporais por conta (velocity), razões de saldo, tempo desde última transação; **o que vaza** (features pós-evento, `isFlaggedFraud`).
   - **Avaliação**: por que ROC-AUC engana com 0,1% de positivos; PR-AUC, recall@precisão fixa, custo esperado; para sobrevivência: c-index, Brier integrado; **split temporal** vs. aleatório e drift.
   - **Calibração e threshold**: Platt/isotônica; escolher threshold por custo de fraude vs. custo de falso positivo.
   - **Deploy**: latência de features em serving, monitoramento de drift, retreino.
3. Escreva `reports/research.md` com esta estrutura fixa (as outras skills leem por seção):
   ```
   # Research — <problema>
   ## TL;DR (5 linhas)
   ## Framing recomendado
   ## Técnicas por etapa (tabela: etapa | técnica | quando usar | evidência/fonte | risco)
   ## Métricas e protocolo de avaliação (o que o gate deve medir e por quê)
   ## Armadilhas de leakage específicas deste dataset
   ## Decisões para este projeto (numeradas, cada uma com a fonte que a sustenta)
   ## Perguntas em aberto para o humano
   ## Fontes (título — URL — ano)
   ```
4. Cada decisão precisa de pelo menos uma fonte. Sem fonte, vai para "Perguntas em aberto", não para "Decisões".
5. Não escreva código. Não altere a SPEC; se a pesquisa contradiz a SPEC, registre em "Perguntas em aberto" e diga isso no resumo final.

## Resumo final (o que volta pro chat principal)
Só o TL;DR e as "Decisões para este projeto". O resto fica no arquivo.
