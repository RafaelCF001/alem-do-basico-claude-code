# src/features — regras locais
- Funções de feature são puras: recebem DataFrame, devolvem DataFrame, sem I/O e sem estado global
- Toda transformação que aprende parâmetros (scaler, encoder) vai dentro do `Pipeline` do sklearn, nunca aplicada antes do split
- Nome de coluna nova: `f_<descrição>`; documente a intuição em uma linha de docstring
