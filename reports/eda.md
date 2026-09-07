# EDA — `data/train.parquet` (PaySim, treino `step ≤ 600`, amostrado)

Números vêm de `reports/eda_profile.json` (gerado por `.claude/skills/eda/scripts/profile.py`).
Apenas `data/train.parquet` foi lido; `data/test.parquet` não foi aberto.

## Shape, tipos, nulos, duplicatas
- Shape: **306.613 linhas × 11 colunas** (treino já downsampled: todos os 6.613 positivos + 300.000 negativos amostrados de 6.252.434, conforme `src/data.py`).
- Tipos: `step` int32, `type` category, `amount`/`oldbalanceOrg`/`newbalanceOrig`/`oldbalanceDest`/`newbalanceDest` float64, `nameOrig`/`nameDest` string, `isFraud`/`isFlaggedFraud` int8 — sem colunas fora do esperado.
- Nulos: **zero** em todas as 11 colunas.
- Duplicatas: **0** linhas duplicadas.

## Desbalanceamento
- `positives = 6.613`, `fraud_rate = 2,157%` **nesta amostra de treino** — não confundir com a taxa real da população (~0,13% citada na SPEC); a taxa aqui é inflada pelo downsample de negativos, então CV e métricas de treino devem ser lidas nesse contexto, mas o teste (intacto) preserva a taxa real.
- Por `type`: fraude só existe em **TRANSFER** (`mean=0,1164`, 3.297/28.330) e **CASH_OUT** (`mean=0,0305`, 3.316/108.863); `CASH_IN`, `DEBIT`, `PAYMENT` têm `mean=0` (0 fraudes em 169.420 linhas somadas). Isso **confirma empiricamente** a afirmação da SPEC ("fraude só existe em type ∈ {TRANSFER, CASH_OUT}").

## Drift temporal
- `fraud_rate_by_time_window` (10 janelas de `step`) varia de **1,14%** a **28,4%** — não é uniforme: janelas `(60.9, 120.8]` (28,4%, n=2.427) e `(420.3, 480.2]` (22,6%, n=2.687) têm volume de transações muito menor que as demais (~40-56 mil) mas fração de fraude bem maior. Isso é esperado: o volume de transações **legítimas** varia ciclicamente por `step` (padrão dia/noite do simulador), enquanto os positivos (não afetados pelo downsample) ficam relativamente concentrados — logo a taxa relativa sobe quando o denominador (negativos) cai.
- Nenhuma das 10 janelas tem `sum=0` — **não existe período sem fraude** dentro do treino.

## Alvo × features
- `amount` (quantis, não média): mediana de fraude **R$ 446.032,95** vs. legítima **R$ 75.387,65** (~5,9×); p90 fraude **R$ 4.256.401** vs. legítima **R$ 366.928**. `amount` é a feature numérica mais correlacionada com o alvo (`abs_corr = 0,2573`, topo de `abs_corr_with_target_top10`).
- `zero_balance_rate`: `newbalanceOrig == 0` em **98,4%** das fraudes vs. **56,8%** das legítimas — assinatura forte de "zerar o saldo de origem". Ao mesmo tempo `oldbalanceOrg == 0` é raríssimo em fraude (**0,5%**) vs. **33,2%** em legítimas — ou seja, contas fraudadas quase sempre chegam com saldo positivo e saem zeradas. `oldbalanceDest == 0` também é mais comum em fraude (**64,8%** vs. **42,4%**).

## Por conta
- `dest_accounts`: 251.940 contas de destino únicas no treino; **6.579** têm ao menos 1 fraude recebida (equivalente aos 6.613 positivos, quase 1:1 conta:fraude); apenas **34** contas têm mais de uma fraude, e o máximo é **2 fraudes por conta**. Não há evidência de "contas mula" com reincidência alta nesta amostra — o padrão dominante é conta comprometida uma única vez, não repetidamente drenada.
- Isso **justifica o framing de sobrevivência como "tempo até a 1ª fraude"** (e não uma contagem repetida de eventos por conta), exatamente como a SPEC define.

## Sobrevivência (`data/survival_train.parquet`)
- `n = 2.676.208` contas, `events = 6.579` (bate com `dest_accounts.with_fraud`), `censoring_rate = 99,754%` — evento extremamente raro, como esperado.
- `km_median = null`: a curva de Kaplan-Meier não cruza 50% de sobrevivência dentro da janela observada (consistente com >99,7% de censura).
- `km_survival_at`: S(24)=0,9983, S(168)=0,99787, S(336)=0,99757 — hazard baixo e relativamente estável ao longo do tempo, sem um "pico" claro de risco logo após a 1ª transação recebida.
- Covariáveis candidatas já calculadas **causalmente** (só transações anteriores ao evento/censura de cada conta): `f_n_received_before`, `f_amount_received_before_mean`, `f_amount_received_before_sum`.

## Leakage
- `leakage_suspects` (script): **apenas `isFlaggedFraud`** — nenhuma outra coluna numérica passou do limiar de correlação 0,95 com o alvo (a maior é `amount` com 0,2573).
- Features que **não podem** entrar no modelo: `isFlaggedFraud` (regra pós-hoc do simulador — decisão #1/leakage da research), `nameOrig`/`nameDest` crus (`FORBIDDEN_FEATURES`), qualquer agregação por conta calculada com `step` **posterior** ao da transação (armadilha #3 da research).
- `newbalanceOrig`/`newbalanceDest` são utilizáveis (resultado imediato da transação, não são "futuro"), mas a research já registrou como pergunta em aberto (#3) que o padrão de "zerar saldo" pode ser um artefato do simulador PaySim — o dado desta EDA (98,4% vs 56,8%) é consistente com essa suspeita: é um sinal forte demais para ser causal em dados reais, então deve entrar como feature com essa ressalva documentada, não como verdade generalizável.

## Qualidade
- `amount_le_0`: **8** linhas com `amount ≤ 0` (0,0026% do treino) — valor impossível para uma transação financeira; tratar (excluir ou clipar) na engenharia de features.
- `constant_columns`: nenhuma (`[]`).

## Hipóteses de features (≤5)
1. **`amount` (bruto ou `log1p(amount)`)** — evidência: maior correlação absoluta com o alvo (0,2573) e mediana de fraude ~5,9× a legítima. Técnica: gradient boosting (research decisão #4) lida bem com a escala bruta; log1p reduz assimetria se usado no baseline logístico.
2. **`f_orig_balance_drained` = (`newbalanceOrig == 0` e `oldbalanceOrg > 0`)** — evidência: 98,4% das fraudes zeram o saldo de origem vs. 56,8% das legítimas. Documentar a ressalva da research (pergunta #3: pode ser artefato do simulador, não necessariamente generalizável a dados reais).
3. **`type` restrito/one-hot a `{TRANSFER, CASH_OUT}`** — evidência: 0 fraudes em `CASH_IN`/`DEBIT`/`PAYMENT` (169.420 linhas). Decisão a registrar na SPEC/`/train`: filtrar o treino a esses dois tipos (reduz ruído, alinhado à prática comum na literatura de PaySim citada na research) ou manter todos os tipos e deixar o modelo aprender o corte via `type`.
4. **`f_n_received_before` / `f_amount_received_before_sum`** (velocity por conta de destino, já calculadas causalmente em `data/survival_train.parquet`) — evidência: research decisão #9 (agregações só com `step` estritamente anterior); podem ser reaproveitadas como features de classificação por transação, não só para sobrevivência.
5. **Janela de `step` (ex.: `step % 24`, bucket cíclico)** — evidência: `fraud_rate_by_time_window` varia de 1,1% a 28,4% entre janelas, indicando padrão cíclico de volume/risco por hora simulada. Deve ser calculada só a partir do `step` da própria transação (sem olhar para trás em outras contas) para não introduzir leakage.

## Implicações para /train
- Split já fixo em `data/{train,test}.parquet` (`step≤600`/`step>600`, treino downsampled 6.613+300.000, teste intacto) — não re-splitar.
- Métrica primária: **PR-AUC** em `data/test.parquet` (taxa real ~0,13%); ROC-AUC só informativo. Baseline obrigatório: regressão logística `class_weight="balanced"`.
- Desbalanceamento: `class_weight`/`scale_pos_weight` primeiro; reamostragem adicional (se usada) só dentro de `imblearn.pipeline.Pipeline`, nunca reaplicando o downsample de `data.py`.
- Testar (e registrar a decisão) filtrar `type ∈ {TRANSFER, CASH_OUT}` vs. manter todos os tipos — 0 fraudes fora desses dois tipos nesta amostra.
- Features proibidas: `isFlaggedFraud`, `nameOrig`/`nameDest` crus, qualquer agregação com `step` futuro; usar `newbalanceOrig`/`newbalanceDest` com a ressalva de possível artefato do simulador.
- Candidato principal: gradient boosting (HistGBM) vs. baseline; sobrevivência via Cox PH sobre `data/survival_train.parquet` (evento raro, censura 99,75%).
