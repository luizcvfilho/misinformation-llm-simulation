# Plano de escrita do TCC orientado por evidências

Este documento concentra a pergunta de pesquisa, o escopo, o histórico metodológico, as evidências internas e externas e a ordem prática de redação do TCC.

## 1. Título e argumento provisórios

**Título de trabalho:** *Deriva informacional em cadeias de reescrita de notícias por agentes baseados em grandes modelos de linguagem*.

**Argumento central:** classificadores binários de veracidade não descrevem adequadamente transformações graduais de conteúdo e enquadramento. O TCC propõe e avalia o STDI como instrumento para observar essas mudanças em cadeias de reescrita condicionadas por perspectivas distintas.

**Limite obrigatório:** o STDI não estabelece se uma notícia é verdadeira ou falsa em relação ao mundo.

**Pergunta de pesquisa provisória:** em que medida a ordem e o tipo de perspectiva atribuída a agentes LLM alteram o conteúdo e o enquadramento emocional de uma notícia ao longo de uma cadeia de reescritas?

## 2. Estrutura recomendada

### 2.1 Introdução

**Objetivo da seção:** apresentar o problema, delimitar deriva versus veracidade, formular pergunta e objetivos e resumir as contribuições.

**Sequência argumentativa:**

1. A circulação de notícias falsas é um problema sociotécnico relevante (`lazer2018science`, `vosoughi2018spread`).
2. A presença de conteúdo gerado e parafraseado por LLMs altera o cenário de detecção (`su2024adapting`, `wu2024sheepdog`).
3. O conteúdo pode se modificar ao longo da transmissão, e não apenas se difundir de forma estática (`zhang2013rumor`, `guo2021truth`).
4. Gerações iterativas por LLMs podem acumular distorções (`mohamed2025broken`).
5. Trabalhos recentes simulam evolução e propagação com agentes LLM (`liu2024skepticism`, `liu2025stepwise`).
6. Lacuna do TCC: observar, de maneira estruturada e interpretável, como diferentes dimensões mudam em cadeias de reescrita sob personas e ordens distintas.

**Não antecipar:** resultados estatísticos finais, pesos como se já estivessem validados ou alegações de causalidade sobre personalidades.

### 2.2 Fundamentação teórica e trabalhos relacionados

#### 2.2.1 Desinformação, fake news e rumores

- Definir os termos usados e escolher uma terminologia consistente.
- Usar `lazer2018science` como visão interdisciplinar.
- Usar `yang2025rethink` para o cenário recente de rumores e LLMs.
- Explicar que propagação, veracidade, crença e transformação textual são objetos diferentes.

#### 2.2.2 Evolução textual e enquadramento

- Apresentar `entman1993framing` para seleção, saliência e enquadramento.
- Construir a linha histórica `zhang2013rumor` → `guo2021truth` → `mohamed2025broken` → `liu2025stepwise`.
- Identificar o ponto de aproximação e a diferença entre FUSE-EVAL e STDI.

#### 2.2.3 Agentes LLM, personas e dinâmica social

- Introduzir redes de agentes com `chuang2024opinion`.
- Discutir consistência e alinhamento de personas com `frisch2024interaction`.
- Comparar o desenho com `liu2024skepticism`, `wang2025echo` e `liu2025mosaic`.
- Usar `chen2025rpa` para critérios de avaliação de role-playing agents.
- Não alegar equivalência entre agentes LLM e pessoas reais.

#### 2.2.4 Detecção de fake news e seus limites para este problema

- Explicar o objetivo tradicional de classificação e a mudança trazida por conteúdo sintético (`su2024adapting`).
- Discutir sensibilidade a estilo e reenquadramento (`wu2024sheepdog`).
- Mostrar por que conhecimento externo é necessário para factualidade (`whitehouse2022knowledge`).
- Concluir que medir deriva e verificar fatos são tarefas complementares, não intercambiáveis.

#### 2.2.5 Representação semântica e dimensão afetiva

- Embeddings e similaridade: `reimers2019sbert`.
- Arquitetura MiniLM: `wang2020minilm`.
- VAD em texto: `buechel2017emobank`, `mohammad2018vad`.
- BERTopic: apresentar apenas se a análise exploratória for efetivamente executada (`grootendorst2022bertopic`, `karnam2026bowling`).

### 2.3 Evolução metodológica do projeto

**Objetivo:** explicar por que o projeto saiu da pergunta binária e chegou ao STDI, sem apresentar um piloto como prova geral de falha dos detectores.

#### Cronologia consolidada para a redação

| Período | Etapa | Evidência e interpretação permitida |
| --- | --- | --- |
| Março de 2026 | Pipeline inicial de coleta, reescrita e auditoria com quatro personas. | Registrar entrada, modelo, persona, prompt e saída. Os modelos e as cotas variaram entre execuções. |
| Abril de 2026 | Teste de classificação binária após reescritas. | Em um recorte com persona conspiratória, houve 25 saídas bem-sucedidas: o detector pré-treinado marcou 0/25 como falsas e a auditoria local sinalizou 1/25. Esse piloto motivou uma medida gradual, mas não demonstra incapacidade geral dos detectores. |
| Abril–maio de 2026 | Primeira versão do STDI, simulação em grafo e análises VAD. | O índice passou a decompor tema, subtópicos, entidades e relações, com contradição interna e VAD como sinais complementares. O piloto VAD mostrou dependência do idioma e do modelo. |
| Agosto de 2026 | Cinquenta pares controlados e comparação entre avaliação semântica por LLM e embeddings/agrupamento. | Os dois métodos foram executados sobre os mesmos pares. As colunas de avaliação humana ainda não foram preenchidas; não apresentar pesos ou limiares como validados por julgamento humano. |
| Agosto de 2026 | Auditoria de domínio e regressões com atributos STDI e TF-IDF. | Usar como diagnóstico de sinais e possíveis atalhos, não como demonstração de detecção factual. |
| Setembro de 2026 | Fórmula vigente e 16 cadeias de quatro posições aplicadas às mesmas 50 notícias. | Foram produzidos 3.199/3.200 passos e 799/800 versões finais com STDI válido. Comparar cenários de forma pareada por notícia e inspecionar exemplos antes de atribuir efeitos às perspectivas. |
| 03/10/2026 | Remoção de `topic_domain` da extração e da regra de deriva temática no comparador `cluster_v2`. | Na execução `simulation_ui_20261003_202828`, 6 dos 30 passos tiveram deriva temática máxima imposta pela diferença de domínio, todos na mesma notícia sobre ICE/DHS. A auditoria em `output/audit/TopicDomainGateAudit_20261003/` preserva os escores históricos e compara a alternativa por similaridade dos temas, mantendo os demais componentes. É análise de sensibilidade, sem validação humana. Novas execuções registram a versão do comparador; os resultados anteriores não foram recalculados automaticamente. |

#### Regras para narrar a evolução

1. Separar a versão do modelo usada em cada resultado; não atribuir retroativamente o modelo atual às execuções iniciais.
2. Apresentar a classificação binária como motivação metodológica, não como prova geral contra detectores.
3. Identificar como históricas as fórmulas anteriores. A fórmula vigente deve ser conferida em `src/misinformation_simulation/topic_drift/metrics.py`.
4. Distinguir extração estrutural por LLM da comparação posterior por embeddings, agrupamento e regras fixas.
5. Distinguir distância ao original, mudança incremental e soma cumulativa.

#### Fontes internas prioritárias

- histórico: `git log --reverse`, especialmente os commits `250142a`, `31c4432`, `963b008`, `1b26c61`, `de91b04`, `7a76eef`, `a50c0c2`, `596b654` e `dd7e118`;
- fórmula e extração atuais: `src/misinformation_simulation/topic_drift/metrics.py` e `src/misinformation_simulation/topic_drift/extraction.py`;
- execuções iniciais e modelos: `output/runs/20260326_*/` a `output/runs/20260413_*/` e `output/execution_report.md`;
- análises VAD: `output/audit/VADPatternAudit/`, `output/audit/TopicMatchedVADAudit/` e `output/audit/VADContextContrastAudit/`;
- avaliação do STDI: `output/stdi_manual_evaluation/` e `output/topic_drift/refreshed_{llm,cluster}/`;
- cobertura e regressões: `output/audit/topic_domain_coverage_1000_description/` e `output/stdi_logistic_regression/`;
- cenários e resultados: `data/graphs/interaction_chains/README.md` e `output/interaction_graph/app_runs/simulation_ui_20260918_000717_*/`.
### 2.4 Materiais e métodos

#### Dados

- Origem, licença, idioma, campos usados e critérios de exclusão.
- Tamanho de cada amostra e relação entre corpus de calibração, regressão e simulação.
- Versão ou hash do conjunto congelado.

#### Geração e agentes

- Modelo e versão, data de execução, temperatura e demais parâmetros.
- Prompt integral ou referência a apêndice.
- Definição operacional de cada persona.
- Ordem das personas e 16 cenários.
- Estratégia de erro, repetição e saídas ausentes.

Referências: `chuang2024opinion`, `frisch2024interaction`, `chen2025rpa`, `abdurahman2025primer`.

#### Extração estrutural e STDI

- Esquema de tema, subtópicos, entidades, relações, enquadramento e contradição.
- Separar extração por LLM da comparação posterior.
- Descrever a fórmula vigente e justificar cada componente.
- Registrar encoder, corpus de ajuste do agrupamento, parâmetros e semente.

Referências: `reimers2019sbert`, `wang2020minilm`, `entman1993framing`.

#### VAD

- Definir valência, arousal e dominância.
- Informar escala, normalização e forma de agregação.
- Discutir a limitação de idioma observada no piloto.
- Documentar a comparação **BERT atual, NRC v2.1 e MEmoLon MTL_grouped** no único fluxo principal `notebooks/simulation_vad_model_comparison_workbench.ipynb`. Os 20 pares sintéticos em inglês usam os mesmos textos e a mesma revisão atual do BERT para os dois léxicos; os resultados conjuntos estão em `output/audit/VADThreeModelContextContrastAudit/`. Escalas nativas: BERT 1–5, NRC −1–1 e MEmoLon 1–9. Para a escala comum 1–5, usar NRC `3 + 2 * valor_nativo` e MEmoLon `(valor_nativo + 1)/2`; depois dividir as diferenças por quatro. Não há validação de português ou superioridade sem julgamento humano.
- A comparação conjunta nas simulações salvas está em `output/audit/SimulationVADModelComparison/`. O recorte usa as mesmas seis cadeias (SSSS, CCCC, PPPP, DDDD, CCPP e PPCC), 50 notícias e 1.200 etapas. Normalizar BERT por `(valor - 1)/4`, NRC por `(valor_nativo + 1)/2` e MEmoLon por `(valor_nativo - 1)/8`; comparar por etapa contra o original e contra a entrada efetiva. Os três usam os textos salvos, sem novas reescritas ou extrações por LLM. O STDI hipotético preserva a contribuição não afetiva histórica e substitui somente VAD após conferir a fórmula e a deriva histórica do BERT. Resumos conjuntos usam as mesmas observações disponíveis para os três; as tabelas conjuntas preservam observações ausentes e disponíveis de cada estimador.
- Explicitar a assimetria de extensão: BERT usa o limite de 512 tokens WordPiece, enquanto MEmoLon percorre o texto inteiro. Registrar a contagem de textos truncados e não atribuir toda divergência exclusivamente ao estimador. A média lexical não trata contexto ou negação; maior cobertura não equivale a validade afetiva.
- **Histórico da decisão em 01/10/2026:** o commit `d09cd838edbd827750933181775efe315c55763f` contém a comparação inicial com NRC v2.1 e BERT histórico. O NRC v2.1 foi retirado da lista de candidatos por decisão do pesquisador, e seus resultados nos dois notebooks foram substituídos por MEmoLon. Suas referências permanecem para rastreabilidade. O NRC v1 e suas traduções de 2022 ficam documentados para possível uso posterior; a tradução portuguesa não constitui nova anotação VAD humana em português. O estimador principal e a fórmula do STDI continuam inalterados.
- **Revisão da decisão em 02/10/2026:** o pesquisador solicitou reunir os três estimadores em um único notebook e, posteriormente, remover pilotos e resultados duplicados. O NRC v2.1 voltou à comparação exploratória, sem mudança no estimador principal. Somente os resultados conjuntos atuais dos pares contextuais e das simulações são mantidos neste fluxo. Por solicitação posterior, preservar também a análise inicial de cobertura bilíngue em `output/audit/PortugueseVADLexiconComparison/`, separada do notebook principal; o relatório e o manifesto registram os textos, as fontes e as limitações. As inferências portuguesas de BERT e NRC v2.1 são sondagens exploratórias, sem validação documentada para esse idioma; cobertura lexical não representa acurácia. A comparação inicial permanece no commit indicado acima. Distinguir o BERT histórico daquela comparação do BERT atual usado na comparação conjunta.
- **Resultados exploratórios desta atualização:** nos 20 pares, a valência positiva superou a negativa em 19 pares com MEmoLon e 20 com BERT; o delta médio foi +0,097 e +0,410, respectivamente, na escala 1–5. Nos 1.200 passos, a deriva média contra o original foi 0,00460 com MEmoLon e 0,01740 com BERT, na escala normalizada 0–1. A cobertura MEmoLon ponderada por tokens foi 100% nos pares sintéticos e 98,25% nas reescritas salvas. Os 1.200 passos passaram na conferência da fórmula histórica; 39 dos 1.204 textos únicos das simulações excederam o limite do BERT. São resultados deste estudo, sujeitos às limitações acima, e não evidências externas de superioridade.

Referências: `buechel2017emobank`, `mohammad2018vad`.
Referências do MEmoLon na comparação atual: `buechel2020memolon`, `buechel2020memolondata`.
Referência da distribuição e traduções do NRC v1: `mohammad2018vadresource`.
Referências do NRC v2 na comparação atual e no histórico: `mohammad2025vadv2`, `mohammad2025breaking`.

#### Validação

- Rubrica dos 50 pares controlados.
- Amostragem predefinida e justificativa.
- Concordância entre anotadores, se houver dois ou mais.
- Correlação/erro entre julgamento humano e componentes calculados.
- Sensibilidade a pesos, prompt, modelo e repetição.

- **Atualização de 06/10/2026:** novas avaliações `dual_stdi_v5` usam três avaliações do judge
  por par por padrão, com quantidade configurável na UI/API/CLI e opção de avaliação única.
  Cada avaliação recebe os mesmos textos e estruturas e reutiliza VAD. A pontuação LLM é a
  média dos STDIs completos de cada avaliação; a média final Dual permanece 50/50 com embeddings.
  Preservar notas, justificativas, respostas brutas, falhas e desvio padrão amostral por componente
  e STDI. Todas as avaliações solicitadas e VAD completo são necessários para uma pontuação
  agregada disponível. Avaliações salvas têm cache por repetição; resultados históricos não
  são recalculados automaticamente. O padrão de três é uma escolha de orçamento, sem validação
  empírica de suficiência. Comparar uma versus múltiplas avaliações em amostra humana predefinida
  antes de alegar melhora. `saha2026judgebudget` fundamenta consultas repetidas para estimar médias
  sob orçamento; seu algoritmo adaptativo não foi adotado. `haldar2025ratingroulette` documenta
  inconsistência entre execuções em outras tarefas, sem validar o STDI deste estudo.

Referências: `abdurahman2025primer`, `chen2025rpa`, `saha2026judgebudget`,
`haldar2025ratingroulette`.

### 2.5 Resultados

Separar em blocos:

1. resultados dos detectores iniciais como motivação;
2. análises VAD exploratórias;
3. validação humana e sensibilidade do STDI;
4. cobertura temática e diagnóstico de atalhos;
5. comparação pareada das cadeias;
6. exemplos qualitativos de baixa e alta deriva, incluindo erros do índice.

Cada tabela deve informar denominador, unidade de análise, cenário, modelo e intervalo de incerteza quando aplicável.

#### Estado exploratório dos resultados

Os números abaixo registram o estado das análises em 22/09/2026 e devem ser recalculados ou conferidos antes da redação final:

- **Regressão e VAD:** a importância por permutação concentrou-se em `vad_arousal` (0,314) e extensão do texto (0,109). Na análise pareada do mesmo tema, o arousal diminuiu em 21 dos 24 pares válidos, com delta médio de −0,196. A reescrita foi solicitada como corretiva, mas não houve checagem factual externa.
- **Cenários:** as médias do STDI final variaram de 0,239 a 0,269. Entre os homogêneos, `PPPP` teve a maior média (0,269) e `DDDD`, a menor (0,239); `SSDD` teve média 0,268. No contraste `CPCP − PCPC`, a diferença pareada foi +0,023 e o intervalo de reamostragem incluiu zero. Esses valores não estabelecem efeito estável da ordem.
- **Domínios:** meio ambiente teve 2 notícias e STDI médio 0,282; governo e políticas públicas, 12 e 0,279; negócios e economia, 33 e 0,249; outros domínios, 3 e 0,217. Os grupos pequenos não sustentam generalização.
- **Saída ausente:** no cenário `DDSE`, a quarta posição da notícia “CAAS defers sustainable aviation fuel levy” produziu a reescrita, mas a extração estrutural retornou JSON inválido. Tratar como falha de processamento, não como resultado substantivo.

Esses valores são resultados deste projeto, não achados da literatura externa.

### 2.6 Discussão

**Perguntas a responder:**

- Quais componentes mudam mais ao longo da cadeia?
- A ordem das perspectivas está associada à deriva final ou incremental?
- Resultados persistem sob métodos ou pesos alternativos?
- Em quais casos a pontuação contradiz o julgamento humano?
- Quanto do efeito pode vir do modelo, do prompt, do comprimento ou do idioma?

Comparar diretamente com `liu2025stepwise` e `mohamed2025broken`. Usar `chuang2024opinion`, `frisch2024interaction` e `abdurahman2025primer` para interpretar ameaças à validade.

### 2.7 Conclusão

- Responder à pergunta de pesquisa no limite dos resultados.
- Apresentar o STDI como contribuição instrumental sujeita à validação realizada.
- Reafirmar que deriva não equivale a falsidade.
- Indicar factualidade externa, outros idiomas, outros modelos e redes com interação como trabalhos futuros.

## 3. Matriz de afirmações e evidências

| Afirmação pretendida | Evidência externa | Evidência interna necessária | Formulação segura |
| --- | --- | --- | --- |
| Informação pode mudar enquanto circula. | `zhang2013rumor`, `guo2021truth` | Exemplos das cadeias | “A literatura modela e observa evolução do conteúdo durante a propagação.” |
| Iteração com LLMs pode acumular distorção. | `mohamed2025broken`, `liu2025stepwise` | STDI por passo e contra o original | “A geração iterativa pode produzir distorção acumulada; investigamos esse fenômeno em reescritas condicionadas por perspectivas.” |
| Personas podem afetar saídas e interações. | `chuang2024opinion`, `frisch2024interaction` | Comparação pareada e checagem de aderência | “O condicionamento por persona está associado a diferenças de comportamento em simulações, mas sua consistência precisa ser avaliada.” |
| Detectores podem explorar sinais de estilo. | `su2024adapting`, `wu2024sheepdog` | Regressão TF-IDF e inspeção de features | “O desempenho pode depender de autoria e estilo, exigindo cautela ao interpretar classificação como veracidade.” |
| STDI mede deriva. | Não pode depender apenas da literatura | Validação humana e sensibilidade | “Neste trabalho, STDI operacionaliza deriva segundo componentes definidos; sua validade é avaliada nos experimentos.” |
| Uma cadeia ou persona causa mais deriva. | Nenhuma fonte substitui o experimento | Desenho pareado, incerteza e controles | Só usar “causa” se o desenho sustentar; caso contrário, escrever “apresentou” ou “esteve associada”. |

## 4. Decisão sobre BERTopic

Não incorporar BERTopic ao cálculo principal antes da validação do STDI. Se houver tempo, executar um estudo exploratório com um único ajuste sobre o corpus completo, análise de estabilidade e revisão humana dos tópicos. O resultado deve ocupar uma subseção de cobertura temática ou um apêndice, e não alterar retroativamente a definição do índice.

## 5. Ordem prática de redação

1. Materiais e métodos, usando o sistema implementado e versões congeladas.
2. Fundamentação e trabalhos relacionados, seguindo a base bibliográfica.
3. Resultados, após completar validação e análises finais.
4. Discussão, conectando resultados e literatura.
5. Introdução, já com contribuição e lacuna estabilizadas.
6. Conclusão, resumo e abstract.

## 6. Checklist antes de considerar uma seção pronta

- Toda afirmação externa verificável tem fonte adequada.
- A fonte realmente sustenta a frase e não apenas trata do mesmo tema.
- Resultados próprios têm caminho de arquivo, amostra e versão identificáveis.
- Pilotos estão marcados como exploratórios.
- Não há equivalência indevida entre deriva, contradição e falsidade.
- Modelos, prompts, parâmetros e datas estão documentados.
- Limitações de idioma e dependência do modelo estão explícitas.
- Chaves citadas existem em `references.bib`.

## 7. Prompt inicial para futuras conversas

> Leia `AGENTS.md`, `docs/template_tcc_ic_ufrj.md`, `docs/base_bibliografica_tcc.md`, `docs/plano_escrita_tcc.md` e `references.bib`. Vamos escrever ou revisar a seção [NOME DA SEÇÃO]. Use apenas referências verificadas, relacione cada citação à afirmação que ela sustenta e sinalize lacunas que ainda exigem fonte ou evidência interna. Não trate STDI como verificação factual.
