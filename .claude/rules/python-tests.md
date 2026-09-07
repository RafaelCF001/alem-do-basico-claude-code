---
paths:
  - "tests/**/*.py"
---

# Regras para testes
- Um comportamento por teste; nome descreve o comportamento (`test_rejects_expired_token`)
- Use `tmp_path` e fixtures; sem mocks de disco ou de rede fora de `conftest.py`
- Teste que só existe pra cobrir linha é ruído: cubra edge cases nomeados no plano
