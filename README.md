# misinformation-llm-simulation

Projeto de simulacao de desinformacao com LLMs.

## Requisitos

- Windows PowerShell
- `uv` instalado
- `make` (opcional, para usar os atalhos do Makefile)

## Setup com uv

No diretorio do projeto:

```powershell
uv venv
uv lock
uv sync
```

Isso cria o ambiente virtual em `.venv`, resolve as dependencias e instala tudo com o proprio `uv`.

## Comandos do Makefile

```powershell
make setup      # cria venv, locka e sincroniza
make sync       # sincroniza dependencias
make lock       # atualiza uv.lock
make add PKG=nome-do-pacote
make notebook   # abre jupyter lab via uv
make clean      # remove .venv
```

## Rodar notebook

```powershell
uv run jupyter lab
```