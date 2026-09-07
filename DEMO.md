# Demo — do zero a um modelo de fraude servido no Databricks (~30 min)

O harness (`.claude/`, `CLAUDE.md`, `SPEC.template.md`, `pyproject.toml`) vem pronto. O código (`src/`, `tests/`)
NÃO vem: o Claude escreve ao vivo, dirigido pelas skills. Isso é o ponto da demo.

## Pré-requisitos (1 dia antes, 30 min)

1. Workspace Databricks com Unity Catalog. Crie/escolha `catalog.schema` onde você tem `USE CATALOG`, `USE SCHEMA`, `CREATE MODEL`.
   Model Serving habilitado na região. Custo: endpoint Small com scale-to-zero, deletado no fim.
2. Autenticação sem token em arquivo de projeto:
   `databricks auth login --host https://<workspace>.cloud.databricks.com --profile demo`
   `export DATABRICKS_CONFIG_PROFILE=demo` no shell que vai rodar `claude`.
   (O kit nega `Read(~/.databrickscfg)` e o sandbox nega leitura do arquivo; o SDK lê pelo perfil sem o Claude ver.)
3. Crie o experimento uma vez: `mlflow experiments create -n /Shared/fraud-serving` (com `MLFLOW_TRACKING_URI=databricks`).
3b. Kaggle: `~/.kaggle/kaggle.json` (o kit nega leitura pelo Claude; `kagglehub` lê sozinho). Baixe o PaySim UMA vez antes — 470 MB — pra não depender da rede na hora. Sem Kaggle: `src/data.py` usa OpenML `creditcard` e a parte de sobrevivência é pulada (diga isso na SPEC).
3c. `/research` usa WebSearch/WebFetch: confira que os domínios do `settings.json` batem com o que sua rede libera. Rode uma vez antes e GUARDE o `reports/research.md` gerado como plano B (pesquisa ao vivo leva 3–5 min e pode variar).
4. Ferramentas locais: `uv`, `jq`, `git`, Claude Code atualizado (`/release-notes`). Linux/WSL2: `bubblewrap socat`.
5. Ensaie o fluxo inteiro UMA vez num diretório descartável. Anote os tempos. O `/tune` com 30 trials de GBR leva ~2 min em laptop; se passar disso, reduza o orçamento na SPEC para 15.
6. Ajuste o allowlist de rede em `settings.json` para o host do seu workspace (`*.cloud.databricks.com` / `*.azuredatabricks.net`).

## Setup da pasta (na hora, 1 min)

```
mkdir fraud-serving && cd fraud-serving && git init
cp -r <kit>/{.claude,CLAUDE.md,SPEC.template.md,pyproject.toml,src} .   # src/ só tem features/ (CLAUDE.md + FORBIDDEN_FEATURES)
uv sync && git add -A && git commit -m "harness"
claude
```
Aceite o trust prompt — ele lista os hooks e allow rules. Comente: "isso é o Claude Code me mostrando o harness antes de ativar".

## Roteiro

### 0. `/context` (1 min)
System prompt + CLAUDE.md. Skills visíveis: `/research /eda /train /evaluate /tune` (descrições, ~300 tokens). Invisíveis: `/pipeline` e `/deploy` (`disable-model-invocation`). Fala: "as que o modelo pode chamar sozinho estão no índice; as que só eu chamo custam zero".

### 1. Spec por entrevista (4 min)
```
Quero um sistema de prevenção a fraudes sobre o PaySim: classificar transações e, por conta de destino, modelar o tempo
até a primeira fraude (sobrevivência). Servir o classificador no Databricks. Me entreviste com AskUserQuestion:
métricas e thresholds, custos de FN/FP, split, o que fica fora de escopo, orçamento. Ao terminar, escreva SPEC.md seguindo SPEC.template.md.
```
Responda rápido (PR-AUC ≥ 0.80, recall@p90 ≥ 0.60, c-index ≥ 0.70, FP = R$15, 25 iterações). `/clear`.

### 2. Plan mode (2 min)
`/plan leia SPEC.md, CLAUDE.md e proponha a estrutura de src/ e tests/ e a ordem das skills. Não implemente.` → `Ctrl+G`, edite uma linha, aprove.

### 3. `/pipeline` — o Maestro (12–15 min, ele roda; você narra)
Ele cria a task list e chama, em ordem:
- **`/research`** (fork, sonnet, `background: false`): 3–5 min de WebSearch/WebFetch. Enquanto roda, mostre o frontmatter: `context: fork`, `allowed-tools` só com Web*, e o contrato "sem fonte → pergunta em aberto". Ao voltar: só TL;DR e Decisões entram no contexto; o resto está em `reports/research.md`. **Este é o slide "Skill + Subagent" ao vivo.**
- **`/eda`** (haiku): mostre o `!`sed…`` puxando só a seção de leakage da research; `profile.py` cospe o JSON; ele interpreta. Aponte: "não abriu test.parquet — a regra da skill".
- **`/train`**: testes de contrato primeiro (vermelho → verde): split temporal, FORBIDDEN_FEATURES, reamostragem dentro do pipeline. MLflow UI: runs `task=classification` e `task=survival`.
- **`/evaluate`**: `gate.py` reprova (logreg balanced fica em PR-AUC ~0.5–0.7 no PaySim; HGB pode passar — se passar de primeira, o `/pipeline` para no gate e você mostra o tune como opcional). Ele **não** mexe no threshold do gate.
- **`/tune`**: `TimeSeriesSplit`, 25 iterações. Gate passa. Resumo final do Maestro: "deploy liberado — rode /deploy".
Se algo reprovar depois do orçamento: o Maestro PARA e reporta. Isso também é demo (goal drift evitado).

### 4. Guardian (3 min)
- `Esc Esc` → rewind pro checkpoint antes do /tune. `/deploy main.ml.fraud fraud-demo` → hook: `passed=false`. Volte.
- `cat ~/.kaggle/kaggle.json` → deny. `@ml-reviewer revise antes do deploy` → contexto limpo, read-only, opus; ele checa research ↔ código.

### 5. `/deploy main.ml.fraud fraud-demo` (4 min)
`ask` rule pede sua aprovação no script. Registro UC + alias Champion + endpoint + `smoke_test.py` (request/response/latência). Diga que o survival foi registrado mas não servido — e por quê.

### 6. Fechamento (2 min)
`/context`, `/usage`. "delete o endpoint" → hook bloqueia → você deleta na mão. Última fala: "o que não pode acontecer não fica em prompt".

## Plano B
- Sem workspace / rede caiu: `export MLFLOW_TRACKING_URI=file:./mlruns`, remova `mlflow.set_tracking_uri("databricks")` do CLAUDE.md, e /deploy vira `--dry-run` (adicione a flag no script antes da demo). O harness inteiro (spec, research, TDD, gate, hooks) roda igual.
- `/research` sem rede: copie o `reports/research.md` que você guardou no ensaio; o `/pipeline` detecta o arquivo e segue.
- PaySim lento: `src/data.py` já downsampleia negativos do treino (SPEC); se ainda lento, use 100k.
- `gate.py` não acha runs: confira o nome do experimento e `DATABRICKS_CONFIG_PROFILE`.
- Sandbox não sobe no Linux: siga sem; as permissões e hooks cobrem a demo.
- Claude "trava": `Esc`, `/clear`, prompt novo — e diga que isso É a boa prática.
