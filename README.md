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

## Auditoria de factualidade apos reescrita (caso com noticias reais)

Se seus dados originais sao todos reais, o fluxo recomendado e verificar se a reescrita contradiz os fatos do texto original.

Para isso, use o notebook:

- `src/bert_fake_real_workbench.ipynb`

Esse notebook:

- recebe pares de texto (`original` e `reescrito`),
- usa um modelo NLI para medir `entailment` e `contradiction`,
- marca cada linha como:
	- `consistente_com_original`, ou
	- `potencialmente_falsa_apos_reescrita`.

### Como rodar

1. Abra o Jupyter:

```powershell
uv run jupyter lab
```

2. Abra `src/bert_fake_real_workbench.ipynb`.
3. Na celula 5 (configuracao), ajuste:

- `PAIRED_FILE`
- `ORIGINAL_COLUMN`
- `REWRITTEN_COLUMN`
- opcionalmente `ROW_ID_COLUMN`

4. Execute todas as celulas do notebook em ordem.

### Saida

- arquivo CSV: `data/rewriting_consistency_audit.csv`
- colunas principais:
	- `entailment`
	- `contradiction`
	- `neutral`
	- `consistency_flag`

### Interpretacao

- `contradiction` alto e `entailment` baixo indicam maior risco de distorcao factual apos reescrita.
- Casos com `consistency_flag = potencialmente_falsa_apos_reescrita` devem ser revisados manualmente.