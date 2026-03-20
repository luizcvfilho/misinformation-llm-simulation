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

## Executar notebooks em sequencia

Use o script [src/run_notebooks_sequentially.py](src/run_notebooks_sequentially.py) para executar notebooks em ordem.

Execucao padrao (carrega `llm_simulation_workbench` e depois `bert_fake_real_workbench`):

```powershell
uv run python src/run_notebooks_sequentially.py
```

Opcoes uteis:

```powershell
# Escolher notebooks e ordem
uv run python src/run_notebooks_sequentially.py --notebooks src/llm_simulation_workbench.ipynb src/bert_fake_real_workbench.ipynb

# Continuar mesmo se um notebook falhar
uv run python src/run_notebooks_sequentially.py --continue-on-error

# Executar no proprio arquivo
uv run python src/run_notebooks_sequentially.py --inplace
```

## Coletar noticias com NewsData.io

O script [src/fetch_newsdata_to_csv.py](src/fetch_newsdata_to_csv.py) busca noticias via API do NewsData.io e salva um novo CSV na pasta [data/](data/).

1. Configure sua chave no `.env`:

```powershell
NEWSDATA_API_KEY=sua_chave_aqui
```

2. Execute o script com `uv`:

```powershell
uv run python src/fetch_newsdata_to_csv.py --output data/newsdata_news.csv --language pt --max-records 200
```

Exemplo com filtros adicionais:

```powershell
uv run python src/fetch_newsdata_to_csv.py --query politica --country br --category politics --language pt --max-records 300 --output data/newsdata_politics_br.csv
```

Argumentos principais:

- `--output`: caminho do CSV de saida (padrao: `data/newsdata_news.csv`)
- `--query`: termo textual para busca (`q` da API)
- `--language`: idioma(s), ex.: `pt` ou `pt,en`
- `--country`: pais(es), ex.: `br` ou `br,us`
- `--category`: categoria(s), ex.: `politics,technology`
- `--max-records`: quantidade maxima de registros para salvar
- `--api-key-env`: nome da variavel de ambiente da chave (padrao: `NEWSDATA_API_KEY`)

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
