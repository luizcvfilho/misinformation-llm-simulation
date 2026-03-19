.PHONY: setup sync lock notebook add clean

setup: ## Cria .venv, gera lockfile e sincroniza dependências
	uv venv
	uv lock
	uv sync

sync: ## Sincroniza o ambiente com pyproject.toml e uv.lock
	uv sync

lock: ## Atualiza uv.lock a partir do pyproject.toml
	uv lock

add: ## Adiciona dependência (uso: make add PKG=nome-do-pacote)
	uv add $(PKG)

notebook: ## Abre o Jupyter Lab usando uv
	uv run jupyter lab

clean: ## Remove ambiente virtual
	powershell -NoProfile -Command "if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }"
