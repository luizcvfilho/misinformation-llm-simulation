# Plano de escrita do TCC orientado por evidências

**Atualização de escopo: 19/10/2026.** Este documento concentra a pergunta de pesquisa, o desenho final, o histórico metodológico e o mapa de evidências para a redação. Orienta a escrita dos capítulos 4–7 a partir das análises existentes e a preparação da versão de entrega.

## 1. Pergunta, argumento e escopo consolidados

**Título de trabalho:** *Deriva informacional em cadeias de reescrita de notícias por agentes baseados em grandes modelos de linguagem*.

**Pergunta de pesquisa:** em que medida a ordem e o tipo de perspectiva atribuída a agentes LLM alteram o conteúdo e o enquadramento emocional de uma notícia ao longo de uma cadeia de reescritas?

**Argumento central:** uma classe binária não descreve as transformações graduais entre versões de um texto. O estudo propõe e avalia o Structured Topic Drift Index (STDI) como instrumento decomponível para observar essas mudanças em trajetórias controladas. As contribuições são o procedimento de geração e comparação, a decomposição do índice e a análise empírica de perspectivas e ordens nos corpora estudados.

**Limite obrigatório:** STDI não estabelece veracidade factual externa, crença, exposição ou compartilhamento humano. Contradição interna, deriva de conteúdo e perfil afetivo são dimensões distintas. VAD não representa todo o enquadramento narrativo: `narrative_frame` é extraído, mas não recebe um componente diretamente ponderado na fórmula.

### Decisões do autor

- As seis rodadas identificadas na Seção 2 são as simulações finais. Não tratá-las como pilotos nem propor novas execuções como condição para concluir o TCC.
- As análises dos resultados já foram realizadas, conforme informado pelo autor. A tarefa de escrita é identificar a evidência de cada conclusão, sem presumir que toda análise sugerida no plano antigo foi executada.
- Não haverá atribuição humana de valores numéricos esperados aos pares. Retirar essa atividade dos objetivos, pendências e requisitos de conclusão. Campos históricos vazios não serão preenchidos artificialmente nem tratados como tarefa necessária.
- Usar comparações quantitativas, casos controlados e inspeções qualitativas existentes. Não foi estabelecida uma referência humana numérica independente; isso limita as conclusões sobre validade quantitativa do índice e não constitui promessa de anotação futura.
- Apresentar somente as oito trajetórias finais. Mencionar cenários anteriores apenas quando necessários para identificar uma análise efetivamente citada, com corpus, modelo e versão próprios. Não apresentar seu catálogo na exposição geral nem apagar artefatos históricos.
- Manter a pergunta central e os sete capítulos. Alinhar objetivos e contribuições ao estudo realizado, sem acrescentar experimentos ao escopo.

## 2. Desenho final e evidências das execuções

### 2.1 Rodadas finais

Todos os diretórios abaixo estão em `output/interaction_graph/app_runs/`:

| Corpus | Gerador | Diretório da execução | STDI | VAD |
| --- | --- | --- | --- | --- |
| Inglês, 50 entradas | `gpt-6-luna` | `simulation_ui_20261016_213859-GPT` | Dual | Dual |
| Inglês, 50 entradas | `grok-build-0.1` | `simulation_ui_20261017_160546-GROK` | Dual | Dual |
| Inglês, 50 entradas | `gemini-3.5-flash-lite` | `simulation_ui_20261017_221432-Gemini` | Dual | Dual |
| Português, 50 entradas | `gpt-6-luna` | `simulation_ui_20261018_172257-GPT-BR` | LLM | LLM |
| Português, 50 entradas | `grok-build-0.1` | `simulation_ui_20261018_172257-GROK-BR` | LLM | LLM |
| Português, 50 entradas | `gemini-3.5-flash-lite` | `simulation_ui_20261019_124942-GEMINIBR` | LLM | LLM |

Os resumos conferidos em 19/10/2026 registram quatro árvores, 50 entradas por árvore e 1.200 reescritas bem-sucedidas por rodada, sem erro de reescrita. Isso documenta cobertura de execução, não qualidade dos textos ou validação do índice. Usar os `*_summary.json` e `*_steps.jsonl` de cada árvore, sem contar simultaneamente nós únicos e suas projeções em trajetórias.

Os 24 resumos e os 7.200 registros de nós únicos documentam essas contagens e os métodos da tabela. Todos os registros apresentam reescrita `success` e STDI selecionado `valid` tanto contra o original quanto incremental. Os hashes dos corpora completos permanecem iguais aos da Seção 2.2. Esses estados finais não comprovam ausência de tentativas anteriores após falha nem permitem contar requisições a partir do número de nós.

**Configurações comuns registradas:** modo `interpretive`, prompt `interpretive_v2`, extração e avaliação estrutural por `chatgpt` / `gpt-6-luna`, três avaliações estruturais por par e VAD por LLM com `gpt-6-luna`. Em inglês, VAD Dual também usa `RobroKools/vad-bert`, e STDI Dual inclui Cluster com `sentence-transformers/all-MiniLM-L6-v2`. Nas rodadas brasileiras, VAD e STDI são LLM; não atribuir a elas BERT ou embeddings para esses cálculos.

**Conferência para o Capítulo 4 em 19/10/2026:** os registros finais identificam a rubrica VAD `llm_vad_expressed_tone_v2`; não atribuir a eles a `v1` descrita no piloto. A conferência dos 7.200 nós não encontrou divergências entre IDs/textos originais salvos e o recorte ordenado dos respectivos CSVs. A primeira redação de Materiais e métodos está em `tcc/elementos-textuais/capitulo-4.tex`, com evidências documentadas em `docs/tcc_chapter_4_evidence.md`. A compilação integrada do projeto foi bem-sucedida. Licenças específicas dos textos e revisão imutável dos pesos locais não foram estabelecidas pelos registros examinados; o capítulo explicita esses limites.

O código corrente identifica `structured_extraction_v5`, `structured_judge_v5`, `cluster_v4` e `dual_stdi_v6`. Conferir as versões preservadas em cada resultado, sem atribuí-las retroativamente às auditorias. O identificador `dual_stdi_v6`, sozinho, não significa que foi selecionado o método Dual.

### 2.2 Corpora e congelamento

| Corpus | Arquivo | Recorte utilizado | Cuidados de descrição |
| --- | --- | --- | --- |
| Político em inglês | `data/graphs/graph_politics_news.csv` | Primeiras 50 notícias após remover `__query_metadata__`; corpo em `description`, título e ID preservados nos registros. | O arquivo contém 100 notícias selecionadas intencionalmente. Explicar o recorte de 50, os critérios em `graph_politics_news.selection.md` e dependências entre coberturas do mesmo evento. São descrições disponíveis, não matérias integrais. |
| Brasileiro em português | `data/elections_2026/simulation_news_100_cleaned.csv` | Primeiras 50 entradas, em ordem; o arquivo contém 100 entradas. | São legendas de publicações de veículos jornalísticos no contexto eleitoral brasileiro. Descrever seleção, limpeza e procedência; `caption_reviewed_not_fact_checked` não significa checagem factual. |

Os mesmos IDs e textos foram preservados entre os três geradores dentro de cada corpus na conferência das rodadas finais. Inglês e português não são traduções dos mesmos itens: variam conteúdo, contexto e origem, além do idioma.

Hashes SHA-256 dos arquivos completos conferidos em 19/10/2026:

| Arquivo | SHA-256 |
| --- | --- |
| `data/graphs/graph_politics_news.csv` | `46fabffefde42ed7a8746d97093f1dab19f5f1524908e4217432aea755cffb64` |
| `data/elections_2026/simulation_news_100_cleaned.csv` | `89582261293f87d494390b4e08e63b5c410b1b1ad2e606fa164d58c351f9abea` |

Conferir a seleção ordenada de 50 entradas contra os textos originais salvos, além do hash do arquivo completo. Documentar origem e licença com evidências disponíveis; presença local não comprova autorização ou representatividade. Essas conferências documentais não exigem novas simulações.

### 2.3 Trajetórias, perspectivas e unidades de análise

Perspectivas finais: direita conservadora (C), esquerda progressista (P), negacionista conspiratória (D) e retransmissora neutra (N, `NeutralRelay`). São instruções computacionais, não categorias psicológicas observadas em pessoas.

| Árvore | Trajetórias | Prefixo compartilhado | Nós únicos por entrada |
| --- | --- | --- | --- |
| N | NNNN, NNDD | NN | 6 |
| C | CCCC, CCPP | CC | 6 |
| P | PPPP, PPCC | PP | 6 |
| D | DDDD, DDNN | DD | 6 |

São oito trajetórias de quatro posições em quatro árvores, com 24 nós únicos por entrada: 1.200 operações lógicas de reescrita por rodada de 50 entradas. A projeção linear contém 1.600 registros de posições porque representa novamente os prefixos compartilhados. Esses números não são contagens de avaliações, tentativas de API, tokens ou custos.

NNNN é referência operacional de retransmissão com alterações mínimas, sem garantia de STDI zero ou de veracidade. NeutralRelay preserva conteúdo, atribuição, enquadramento e grau de certeza recebidos; não reconstrói automaticamente o original. A persona cética S, quando aparecer em uma análise anterior, não equivale a N.

CCPP/PPCC e DDNN/NNDD mantêm composição, mas mudam posições e a persona final; não isolam seu efeito. Trajetórias com prefixo compartilhado contêm as mesmas saídas iniciais. Comparar de forma pareada por entrada, preservando essa dependência. Posições, trajetórias projetadas, notas do avaliador e resultados de cache não são gerações independentes.

Configurações: `data/graphs/branching/neutral_relay/`, `neutral_relay_grok/`, `neutral_relay_gemini/` e `docs/branching_interaction_graphs.md`. As versões lineares são referências de configuração, não a topologia das rodadas finais.

## 3. Estrutura recomendada e conteúdo de cada capítulo

### 3.1 Introdução

**Função:** apresentar problema, lacuna, pergunta, objetivos e contribuições compatíveis com o estudo realizado.

1. Contextualizar notícias falsas e diferenciar difusão de transformação (`lazer2018science`, `vosoughi2018spread`).
2. Situar reescrita por LLM e sensibilidade dos detectores à geração e ao estilo (`su2024adapting`, `wu2024sheepdog`).
3. Introduzir evolução de conteúdo e geração iterativa (`zhang2013rumor`, `guo2021truth`, `mohamed2025broken`, `liu2025stepwise`).
4. Delimitar o STDI como descrição de trajetórias e componentes, sem checagem factual externa.
5. Explicitar três geradores, dois corpora e sequências de perspectivas no escopo, remetendo parâmetros ao capítulo 4.

Os objetivos devem cobrir geração, extração, distância ao original, mudança incremental, VAD e comparações pareadas. A avaliação deve corresponder às comparações quantitativas e inspeções qualitativas realizadas. Retirar notas humanas e não prometer sensibilidades a pesos ou prompts sem resultados correspondentes. A organização do trabalho deve anunciar avaliação e análise, sem protocolo de anotação humana.

Não antecipar resultados estatísticos finais, afirmar validade geral dos pesos ou atribuir causalidade a personalidades humanas.

### 3.2 Fundamentação teórica e trabalhos relacionados

| Eixo | Fontes prioritárias | Limite de interpretação |
| --- | --- | --- |
| Desinformação, notícias falsas e rumores | `lazer2018science`, `yang2025rethink`, `vosoughi2018spread`. | Separar veracidade, crença, exposição, compartilhamento e transformação textual. |
| Evolução e enquadramento | `zhang2013rumor`, `guo2021truth`, `entman1993framing`, `mohamed2025broken`, `liu2025stepwise`. | Comparar FUSE-EVAL e STDI pelo alvo e desenho; VAD não esgota enquadramento narrativo. |
| Agentes e personas | `chuang2024opinion`, `frisch2024interaction`, `liu2024skepticism`, `wang2025echo`, `liu2025mosaic`, `chen2025rpa`. | Aderência não é garantida; simulação não equivale a pessoas ou plataformas reais. |
| Detectores e limites | `su2024adapting`, `wu2024sheepdog`, `whitehouse2022knowledge`. | Deriva e checagem factual são diferentes; pilotos próprios não são benchmarks factuais. |
| Representação semântica | `reimers2019sbert`, `wang2020minilm`. | MiniLM pertence ao caminho Cluster, não ao comparador LLM das rodadas brasileiras. |
| Dimensão afetiva | `buechel2017emobank`, `mohammad2018vad`, `buechel2020memolon`, `buechel2020memolondata`, `mohammad2018vadresource`, `mohammad2025vadv2`, `mohammad2025breaking`. | Distinguir modelo neural, normas lexicais e LLM; disponibilidade em português não comprova validade no corpus. |
| Estimador neural utilizado | `robrokoolsvadbert`, ficha oficial do modelo. | Documenta BERT para regressão VAD e o idioma inglês; não é validação em português nem referência humana de verdade. A data de publicação não foi informada. |
| Avaliação por LLM e repetição | `saha2026judgebudget`, `haldar2025ratingroulette`, `abdurahman2025primer`. | Repetição descreve variabilidade, sem validar pesos ou acurácia; três avaliações são decisão de orçamento. |

Conferir a seção da fonte primária antes de associá-la a uma afirmação nos capítulos. A literatura pode recomendar validação humana sem que ela tenha sido realizada neste estudo; explicitar a diferença. Fórmulas e agregação pertencem ao capítulo 4. BERTopic só aparece se sustentar uma análise efetivamente usada, sem incorporação ao STDI.

### 3.3 Evolução metodológica do projeto

**Função:** reconstruir decisões que levaram ao desenho final, sem catálogo de desenhos descartados ou prova geral contra detectores.

| Período | Etapa | Evidência e uso na redação |
| --- | --- | --- |
| Março–abril de 2026 | Coleta, reescrita e auditorias binárias em diferentes conjuntos e geradores. | `output/runs/20260326_*/` a `output/runs/20260413_*/`, `output/execution_report.md` e Git. Relacionar matriz e recortes adicionais do capítulo 3; conferir denominadores sem somar saídas sobrepostas. A resposta binária motivou descrição gradual, sem incapacidade geral dos detectores. |
| Abril–maio de 2026 | STDI, grafo e análises afetivas. | Histórico de `topic_drift/metrics.py` e extração; auditorias `VADPatternAudit/`, `TopicMatchedVADAudit/`, `VADContextContrastAudit/`. Separar fórmulas históricas e contradição interna de falsidade. |
| Agosto de 2026 | Comparação dos mesmos 50 pares por LLM e embeddings/agrupamento; cobertura e regressão. | `output/stdi_manual_evaluation/`, `output/topic_drift/refreshed_llm/`, `refreshed_cluster/`, `output/stdi_logistic_regression/`. Não foram estabelecidas notas humanas numéricas, e essa anotação não será realizada. Regressões distinguem originais/reformulações, não classes factuais verificadas. |
| 01–02/10/2026 | Alternativas ao VAD inicial em português e inglês. | Auditorias bilíngues e comparações BERT/NRC/MEmoLon. O pesquisador não considerou os substitutos satisfatórios para o uso pretendido; sustentar a escolha nos diagnósticos, sem superioridade geral. |
| 03/10/2026 | Remoção da regra temática de domínio no `cluster_v2`; modo interpretativo. | `TopicDomainGateAudit_20261003/`: 6/30 passos de `simulation_ui_20261003_202828` receberam deriva temática máxima pela diferença de domínio. Sensibilidade histórica, com resultados preservados; `config/prompts.py` documenta permissões de transmissão. |
| 04–05/10/2026 | Inspeção qualitativa e casos controlados estruturais. | `RewriteQualitativeAudit_20261004/` e `STDIControlledInformationAudit_20261005/`. Omissões, motivos/certeza e limites do índice; qualificadores, polaridade e valores numéricos. Conferir versões: ampliação numérica posterior não foi validada automaticamente pelo teste anterior. |
| 06/10/2026 | Piloto VAD BERT/LLM; métodos selecionáveis e repetição do avaliador. | `LLMVADComparison_20261006/`, `docs/dual_stdi.md`, `docs/vad_stdi_methods.md`. Explicar alternativa por LLM e configurações finais por corpus. |
| 07–09/10/2026 | NeutralRelay e árvores com prefixos compartilhados. | `data/graphs/branching/`, `docs/branching_interaction_graphs.md`. Explicar referência neutra e dependência, sem catálogo anterior. |
| 14–19/10/2026 | Proveniência, cache, recuperação e conclusão das seis rodadas finais. | Resumos e registros da Seção 2. Registrar configurações efetivas; interface só entra quando necessária à rastreabilidade. Não anunciar nova rodada como pendência. |

Explicar a passagem de reescrita fiel a retransmissão interpretativa: o modo final admite omissões e interpretações de motivo ou causalidade, mas restringe invenção de detalhes concretos. Registros finais indicam ausência do título separado tanto na reescrita quanto na extração; não transferir a assimetria de auditorias anteriores ao experimento final.

Identificar cadeias antigas somente no recorte exploratório que as exige. Não transportar escores entre versões ou substituir S por N.

**Proveniência dos diagnósticos iniciais:** os totais de 2.327 saídas no detector e 2.281 pares/51 sinalizações na auditoria provêm dos CSVs `audit/PreTrainedAudit/pretrained_fake_news_summary.csv` e `audit/LocalAudit/audit_summary.csv` das 12 execuções abaixo, em `output/runs/`:

| Conjunto | Conservadora | Progressista | Conspiratória | Cética |
| --- | --- | --- | --- | --- |
| Geral | `20260326_235646` | `20260327_082745` | `20260327_235633` | `20260328_120949` |
| Ciência | `20260329_010133` | `20260329_183930` | `20260329_230134` | `20260330_095024` |
| Política | `20260331_000153` | `20260331_073343` | `20260331_232329` | `20260401_073809` |

A execução parcial de ciência/progressista `20260329_111102` registrou 143 saídas e não entra na agregação; foi selecionada a posterior `20260329_183930`, da mesma condição. As rodadas adicionais de política são `20260413_223616`, `20260413_224624`, `20260413_225416` e `20260413_230539`: 400 saídas, 392 pares e cinco sinalizações. Em todos esses recortes, o detector pré-treinado atribuiu zero classes falsas.

O recorte de 25 saídas do plano anterior é outra execução, registrada em `output/execution_report.md` em 14/04/2026, com cinco notícias gerais, perspectiva conspiratória e seis configurações: uma falhou nas cinco solicitações e cinco produziram 25 saídas válidas. O detector avaliou 25 saídas e atribuiu zero classes falsas; a auditoria processou 25 pares e sinalizou um. Os CSVs agregados dessa execução estão em `output/audit/PreTrainedAudit/` e `output/audit/LocalAudit/`, fora dos diretórios individuais das rodadas anteriores. Não confundir esse ensaio com uma amostra de 25 notícias independentes ou com a matriz principal.

### 3.4 Materiais e métodos

#### Dados, geração e reprodutibilidade

- Documentar recortes da Seção 2, origem, seleção/limpeza, campos, ordem, IDs, hashes, licença e limitações dos textos.
- Identificar gerador e avaliadores separadamente, com modelos, provedores, datas, prompts e versões salvas.
- Descrever perspectivas, modo interpretativo e oito trajetórias em quatro árvores.
- Registrar parâmetros efetivos: o wrapper solicita temperatura 0,8 para Grok/Gemini e omite temperatura para GPT-6. Parâmetro solicitado nos metadados não comprova igualdade de amostragem.
- Explicar recuperação, cache e ausências. Reutilização não é nova geração nem avaliação independente. Separar requisições, tentativas, operações lógicas e projeções.

Referências: `chuang2024opinion`, `frisch2024interaction`, `chen2025rpa`, `abdurahman2025primer`. Fontes internas: `docs/api_costs_and_cache.md`, `docs/llm_error_recovery.md`, resumos e proveniência.

#### Extração e STDI

- Distinguir extração LLM e comparação posterior. Descrever tema, subtópicos, entidades, relações, asserção, polaridade, valores numéricos, enquadramento e contradição interna. Nem todo campo extraído tem peso próprio.
- Identificar uso de Cluster, LLM e Dual. No Cluster, registrar encoder, corpus do agrupamento e regras de alinhamento/qualificadores. O avaliador LLM usa textos e estruturas com rubrica própria, sem checagem factual externa.
- Manter quatro pesos de conteúdo de 0,25 e incrementos de contradição/VAD, cada um com força 0,20 da distância restante. São convenções do instrumento, sem calibração por anotadores.

Forma operacional, com C como conteúdo, K como contradição e A como deriva VAD:

```text
C = (theme + subtopic + entity + relation) / 4
J = C + (1 - C) * 0.20 * K
F(S, A) = J + (1 - J) * 0.20 * A
A_dual = (A_model + A_llm) / 2
STDI_dual = (F(S_cluster, A_selected) + mean_j F(S_llm_j, A_selected)) / 2
```

As três avaliações estruturais reutilizam textos, estruturas e VAD. Promediar STDIs completos, sem aplicar a fórmula aos componentes médios. Em Dual/Dual, os ramos exibidos usam VAD Modelo e VAD LLM, respectivamente; o total reaplica VAD Dual aos dois ramos antes da média 50/50. A média dos ramos exibidos pode diferir do total final.

Separar a quantidade configurada de avaliações das requisições efetivas: cada avaliação tem seu tratamento de tentativas após falha, pode reutilizar uma resposta compatível e pode ser dispensada pelo atalho registrado para textos idênticos. O cache preserva o resultado de uma operação anterior; não gera uma réplica independente. Conferir essas situações na proveniência, sem converter três avaliações solicitadas em três chamadas novas obrigatórias por par.

Preservar contagens solicitadas/válidas, notas, justificativas, respostas e desvio padrão amostral. Repetição não equivale a repetir extração, VAD ou geração, nem comprova suficiência de três notas. Ausências não são zeros; não preencher um método com escores de outro. O agregado exige os componentes requeridos pela configuração. Desacordo entre métodos não é confiança calibrada.

Separar distância ao original, mudança contra o pai efetivo e soma cumulativa dos incrementos. A soma pode exceder 1 e não equivale à distância final; registrar completude e incrementos válidos.

#### VAD: escolha e configuração final

Narrar a escolha no capítulo 3, os fundamentos no capítulo 2 e os procedimentos nesta seção. O modelo inicial foi comparado a alternativas lexicais e a um LLM para português e inglês. Conforme a decisão do pesquisador, os candidatos lexicais não justificaram substituir o estimador inicial; as limitações nos testes motivaram o VAD por LLM. Português usa LLM, e inglês conserva BERT junto ao LLM. A escolha não demonstra superioridade universal ou validade em português do BERT.

| Evidência em `output/audit/` | Configuração e recorte | Interpretação permitida |
| --- | --- | --- |
| `PortugueseVADLexiconComparison/` | NRC v1, NRC v2.1, MEmoLon e BERT em legendas/controles bilíngues. | Cobertura, idioma, negação e contexto. NRC v2.1 usa léxico inglês nas sondagens portuguesas; coincidência lexical não valida o idioma. |
| `VADThreeModelContextContrastAudit/` | BERT, NRC v2.1 e MEmoLon nos mesmos 20 pares sintéticos em inglês. | Comparação direcional sem padrão humano numérico de notícias. Distinguir revisão atual e histórica do BERT. |
| `SimulationVADModelComparison/` | Três estimadores em textos salvos de seis cadeias anteriores: SSSS, CCCC, PPPP, DDDD, CCPP, PPCC; 50 notícias e 1.200 etapas. | Sensibilidade à troca de VAD sobre textos fixos. As cadeias são identificadas apenas para explicar este recorte, sem compor o desenho final. |
| `LLMVADComparison_20261006/` | 75 textos únicos em inglês: 20 pares sintéticos e 30 pares original/final das mesmas cinco notícias em seis cadeias; avaliação individual por LLM. | Diferenças BERT/LLM e justificativas inspecionáveis. As cinco notícias são as unidades independentes do recorte de simulação. |

Na comparação lexical, escalas nativas: BERT 1–5, NRC v2.1 −1–1, MEmoLon 1–9. Para 1–5, NRC `3 + 2 * valor_nativo`, MEmoLon `(valor_nativo + 1)/2`; para 0–1, BERT `(valor - 1)/4`, NRC `(valor_nativo + 1)/2`, MEmoLon `(valor_nativo - 1)/8`. Converter amplitude não calibra estimadores. Média lexical não trata contexto ou negação, e português disponível não equivale a anotações humanas de documentos nesse idioma.

BERT usa até 512 tokens WordPiece; léxicos percorrem o texto inteiro e o LLM recebe texto completo conforme seu protocolo. Na comparação exploratória das simulações, 39 dos 1.204 textos únicos excederam o limite BERT; no piloto BERT/LLM, cinco reescritas finais excederam esse limite. Não atribuir divergências só ao estimador, ignorando extensão processada.

No cálculo final, cada estimador fornece V, A e D em escala nominal 1–5. Deriva por dimensão: `min(abs(rewrite - reference)/4, 1)`; deriva VAD: média das três dimensões. VAD Dual promedia derivas normalizadas, nunca vetores brutos antes da distância. O LLM avalia tom expresso em um texto, sem seu par, persona ou nota BERT. Isso não mede emoção do leitor, veracidade ou impacto comportamental.

O piloto de 06/10 registrou deriva média de 0,01986 com BERT e 0,07167 com LLM nos 30 pares de simulação; nos 20 controles, 0,05450 e 0,06250. São resultados exploratórios desse recorte. Maior deriva não demonstra superioridade; a interpretação salva também registra possível mistura entre agência do ator e do narrador em dominância.

Preservar a história de 01–06/10 sem transportar ao desenho final afirmações antigas de que o estimador principal não mudou ou de que não havia simulações novas. Mudanças operacionais de notebook só entram no texto se explicarem uma decisão científica.

#### Avaliação e análise adotadas

A avaliação usa comparações quantitativas e inspeções qualitativas documentadas. Os 50 pares históricos, a auditoria posterior de 36 pares primários e os 18 casos diagnósticos de desenvolvimento/teste são conjuntos distintos. Não somá-los como conjunto anotado nem chamar expectativas sintéticas de padrão humano validado.

A auditoria corrigida de 05/10 produziu escores completos em 36/36 pares primários nos dois caminhos. Nos testes de ordenação, Cluster atendeu 6/9 expectativas e LLM, 9/9, naquela versão. Isso não é acurácia validada nem comprova o desempenho da configuração final; versões e expectativas diagnósticas devem acompanhar os números.

Os campos humanos dos 50 pares permanecem vazios por decisão de escopo. Não calcular correlação/erro contra notas inexistentes, concordância entre anotadores ou alegar calibração de pesos. Inspeção pelo pesquisador não será apresentada como anotação quantitativa independente.

Nas rodadas finais, comparar cenários e geradores de forma pareada por entrada dentro de cada corpus. Identificar a análise salva antes de citar médias, intervalos e sensibilidades. Não assumir que todas as análises sugeridas anteriormente foram realizadas. Conferir unidade de reamostragem e dependência entre trajetórias ao citar intervalos.

Entre corpora, distinguir idioma/conteúdo e método: inglês usa STDI/VAD Dual; português, LLM/LLM. O ramo LLM salvo em inglês pode oferecer procedimento semelhante ao português, mas não elimina diferenças de corpus. A comparação não isola efeito linguístico.

### 3.5 Resultados

Organizar em blocos, priorizando as rodadas finais:

**Primeira redação implementada em 19/10/2026:** `tcc/elementos-textuais/capitulo-5.tex`, com sete tabelas geradas em `capitulo-5-tabelas.tex` e 14 figuras vetoriais em `tcc/figuras/chapter5/`. A análise reproduzível está em `scripts/analyze_tcc_chapter_5.py`, com agregados e manifesto em `output/analysis/tcc_chapter_5/` e rastreabilidade em `docs/tcc_chapter_5_evidence.md`. O Capítulo 4 documenta os procedimentos adicionais. As configurações e os escores originais foram preservados, sem novas chamadas a modelos.

O capítulo inclui evolução contra o original, incrementos e variação do afastamento; boxplots finais; contrastes pareados de continuações, sequências e geradores; decomposição e direção VAD; comparação Cluster/LLM em inglês com VAD próprio e comum; comparação dos corpora com LLM/LLM; indicadores lexicais de números, atribuição e incerteza, extensão e contagens de entidades/relações; e seis casos qualitativos. As estatísticas são descritivas, sem testes de hipótese ou intervalos populacionais. Os indicadores lexicais são aproximações exploratórias e não constituem anotação validada. Uma oscilação de STDI em versões lexicalmente equivalentes foi registrada como limite de mensuração, sem modificar os resultados salvos.

1. Diagnósticos dos detectores como motivação, sem repetir extensamente o histórico.
2. Comparação exploratória e escolha de VAD, separadas das simulações finais.
3. Avaliação STDI por casos controlados, comparação de métodos e inspeções qualitativas, substituindo a seção prevista de validação humana.
4. Cobertura e regressões somente quando sustentarem diagnóstico relevante, identificando grupos e sem interpretação como checagem factual.
5. Cobertura de execução e comparação pareada das oito trajetórias nas seis rodadas finais, distinguindo gerador, corpus, STDI e VAD.
6. Casos qualitativos que expliquem escores, mudanças, convergências e limites do índice.

Cada tabela deve informar corpus, denominador, unidade, configuração e versão, além de incerteza quando disponível e adequada. Não inventar intervalos, testes ou métricas ausentes das análises. Não transportar médias históricas às tabelas finais.

### 3.6 Discussão

Perguntas orientadoras, a responder com as análises existentes:

- Quais componentes mudam mais e como distância ao original e mudança incremental descrevem trajetórias diferentes?
- Como cenários/geradores diferem dentro de cada corpus? O que a comparação de ordem permite afirmar, considerando composição, posições e persona final?
- Onde métodos e estimadores VAD divergem? Quais mudanças qualitativas ficam mal representadas no escore?
- Como extração, rubricas, extensão, seleção dos corpora e falta de referência humana numérica limitam as conclusões?

Discutir prefixos compartilhados, cache, falta de repetições independentes da geração, parâmetros distintos dos provedores e avaliador comum. `gpt-6-luna` gera uma condição e avalia todas: possível dependência metodológica, sem demonstração automática de favorecimento. A variação das três notas não expressa a incerteza de toda a geração e mensuração.

Comparar alvo e limites com `liu2025stepwise` e `mohamed2025broken`. Usar `chuang2024opinion`, `frisch2024interaction`, `abdurahman2025primer` e fontes de avaliação por LLM para contextualizar limitações, sem atribuir à literatura resultados próprios.

### 3.7 Conclusão

- Responder à pergunta no alcance das seis rodadas e análises realizadas.
- Apresentar contribuição instrumental com avaliação quantitativa/qualitativa delimitada, sem validação humana numérica ou validade geral.
- Reafirmar deriva versus falsidade e limites de generalização para pessoas/plataformas.
- Factualidade externa, novos corpora, idiomas além dos estudados, modelos além dos três utilizados e interação social podem ser extensões. Português e comparação de geradores já foram experimentados e não são novidades futuras.

## 4. Matriz de afirmações e evidências

| Afirmação ou decisão | Evidência externa | Evidência interna | Formulação permitida |
| --- | --- | --- | --- |
| Informação pode mudar na transmissão/geração iterativa. | `zhang2013rumor`, `guo2021truth`, `mohamed2025broken`, `liu2025stepwise`. | Trajetórias e exemplos salvos. | A literatura motiva observar mudanças; o estudo examina reescritas por perspectivas. |
| Personas estão associadas a diferentes saídas. | `chuang2024opinion`, `frisch2024interaction`, `chen2025rpa`. | Comparações pareadas e inspeções. | Diferenças observadas, sem equivalência com pessoas ou causalidade isolada da persona. |
| Classe binária não localiza mudanças graduais. | `su2024adapting`, `wu2024sheepdog` contextualizam geração/estilo. | Auditorias e textos iniciais. | Motivar outra unidade de análise, sem inferir falha geral dos detectores. |
| STDI operacionaliza dimensões de deriva. | Fontes fundamentam componentes, não validam o índice por si só. | Fórmula, casos controlados, métodos e inspeções. | Explicitar definição/limites, sem calibração por notas humanas inexistentes. |
| Testes VAD motivaram o uso de LLM. | Fontes dos recursos definem métodos/escalas. | Auditorias, exemplos e decisão do pesquisador. | Candidatos não justificaram substituir o modelo inicial; adotou-se LLM em português e BERT/LLM em inglês, sem superioridade universal. |
| Repetir notas descreve variabilidade. | `saha2026judgebudget`, `haldar2025ratingroulette`. | Notas e estatísticas salvas. | Três notas são escolha de orçamento, sem equivalência a gerações ou referência humana. |
| Corpora em idiomas distintos apresentam diferenças. | Nenhuma fonte substitui controle experimental. | Corpora/métodos da Seção 2. | Descrever condições observadas, sem isolar diferenças como efeito do idioma. |

## 5. Mapa de evidências para a redação

| Uso | Arquivos prioritários | Conferência ao citar |
| --- | --- | --- |
| Resultados finais | Seis diretórios da Seção 2, `*_summary.json`, `*_steps.jsonl`, trajetórias em `paths/`. | Identificar árvore, entrada, nó, método e análise; não duplicar prefixos. |
| Corpus inglês | `data/graphs/graph_politics_news.csv`, `data/graphs/graph_politics_news.selection.md`. | Seleção de 100 versus recorte de 50; contexto efetivo nos registros. |
| Corpus brasileiro | `data/elections_2026/simulation_news_100.csv`, `simulation_news_100_cleaned.csv`, `notebooks/elections_2026_eda.ipynb` e proveniência selecionada. | Foi usada a versão limpa; não presumir que o notebook documenta toda edição editorial do CSV. |
| Primeiros diagnósticos | `output/execution_report.md`, `output/runs/`, Git. | Relacionar recorte inicial, matriz e rodadas adicionais, sem sobreposição. |
| Comparação inicial STDI | `output/stdi_manual_evaluation/scored_stdi_pairs.csv`, `output/topic_drift/refreshed_llm/comparison_results.csv`, `refreshed_cluster/comparison_results.csv`. | Campos humanos vazios não são observações; conferir fórmula/extração históricas. |
| Regra de domínio | `output/audit/TopicDomainGateAudit_20261003/summary.json`, `domain_gate_sensitivity.csv`. | Sensibilidade histórica, sem recálculo automático dos escores. |
| Inspeção e casos controlados | `output/audit/RewriteQualitativeAudit_20261004/findings.md`, `review_texts.md`; `output/audit/STDIControlledInformationAudit_20261005/findings.md`, `scored_pairs.csv`, `diagnostic_order_checks.csv`, `evidence_manifest.json`. | Distinguir texto, extração e comparador; informar versões e expectativas diagnósticas. |
| Casos adicionais | `data/synthetic/stdi_information_change_pairs.json`, `data/synthetic/dual_stdi_validation_pairs.json`. | Casos/comandos existentes não comprovam execução ou anotação humana. |
| Escolha VAD | Quatro auditorias da Seção 3.4, `findings.md`, manifestos, `LLMVADComparison_20261006/interpretation.md`. | Idioma, escala, extensão, recorte/modelo; separar cobertura, sensibilidade e acurácia. |
| Regressão | `output/stdi_logistic_regression/`, `output/audit/topic_domain_coverage_1000_description/`, `output/audit/TopicMatchedVADAudit/`. | Reescritas corretivas não foram verificadas externamente; desempenho não é detecção factual. |
| Definição operacional | `src/misinformation_simulation/topic_drift/metrics.py`, `structured_comparison.py`, `dual_workflow.py`, `extraction.py`, `src/misinformation_simulation/config/prompts.py`, `docs/dual_stdi.md`, `docs/vad_stdi_methods.md`. | Seleção/proveniência de cada rodada prevalecem sobre padrões atuais. |

Números e interpretações das auditorias são resultados deste estudo, não da literatura. Se uma análise informada pelo autor ainda não estiver identificada, localizar seu artefato antes de redigir a conclusão; não presumir necessidade de repetir experimentos ou produzir notas esperadas.

## 6. Ordem prática e pendências de redação

1. Revisar a primeira redação de Materiais e métodos, já desenvolvida com dados, parâmetros, modelos e versões salvos.
2. Revisar a primeira redação de Resultados e suas evidências, já implementadas com diagnósticos históricos separados das seis rodadas finais.
3. Redigir Discussão com o alcance das comparações e as limitações do desenho e da mensuração.
4. Redigir Conclusão, respondendo à pergunta de pesquisa no alcance dos resultados apresentados.
5. Preparar resumo, abstract e elementos institucionais, conferindo a coerência e a apresentação da versão de entrega.

Pendências da redação: localizar o artefato de cada análise citada; detalhar origem, licença e preparação dos corpora; documentar modelos, parâmetros e versões; consultar fontes primárias para frases novas. Preservar os denominadores e os recortes históricos identificados neste plano e conferir outros recortes caso sejam acrescentados. Manter as bibliografias sincronizadas quando houver mudanças e realizar a conferência visual final. Não aguardar notas humanas numéricas ou novas simulações, nem converter tarefas documentais em promessas de experimentos ou sensibilidades não executadas.

BERTopic só entra se houver análise existente e pertinente. Não condicionar o término à sua execução nem alterar retroativamente o STDI.

## 7. Checklist antes de considerar uma seção pronta

- Afirmações externas têm fontes que sustentam a frase específica.
- Resultados próprios têm arquivo, corpus, amostra, unidade e versão identificáveis.
- Rodadas finais estão separadas de diagnósticos, sem catálogo de desenhos descartados.
- Objetivos correspondem às análises realizadas, sem promessa de notas humanas.
- Limitações não são tarefas futuras obrigatórias nem são omitidas como se houvesse validação independente.
- Deriva, contradição, enquadramento, VAD e veracidade estão diferenciados.
- Gerador, extração, comparação e VAD estão identificados por corpus.
- Prefixos, notas repetidas e cache não são gerações independentes.
- Comparações entre corpora não isolam indevidamente o idioma.
- Pesos/faixas/escores não são apresentados como calibrados contra referência humana inexistente.
- Chaves citadas existem em `references.bib`; conferir a cópia LaTeX na revisão dos capítulos.

## 8. Prompt inicial para futuras conversas

> Leia `AGENTS.md`, `docs/template_tcc_ic_ufrj.md`, `docs/base_bibliografica_tcc.md`, `docs/plano_escrita_tcc.md` e `references.bib`. Vamos escrever a seção [NOME]. As seis rodadas são finais; não haverá notas humanas numéricas esperadas. Use análises existentes e identifique evidências. Preserve VAD LLM em português e BERT/LLM em inglês. Mencione cenários antigos somente quando necessários a uma análise citada. Não trate STDI como verificação factual. Informe os arquivos e caminhos para reprodução no Overleaf.

## 9. Alterações para replicar no Overleaf

Ao transferir alterações de redação, substituir os arquivos correspondentes de `tcc/elementos-textuais/capitulo-*.tex` em `elementos-textuais/capitulo-*.tex` no Overleaf. Preservar a estrutura de pastas e as inclusões de `main.tex`, salvo mudança necessária na organização do trabalho.

A base canônica `references.bib` corresponde a `elementos-postextuais/referencias.bib` no Overleaf e à cópia local `tcc/elementos-postextuais/referencias.bib`. Ao alterar referências, manter as três versões correspondentes sincronizadas.

A cópia de trabalho `tcc/formatacao.tex` usa `\usepackage{graphicx}` para permitir seleção automática do driver. Se a cópia do Overleaf ainda contiver `\usepackage[pdftex]{graphicx}`, substituir apenas essa linha em `formatacao.tex`. Manter pdfLaTeX como compilador recomendado e preservar o template canônico.

Após transferir fontes ou referências, recompilar o projeto e atualizar a bibliografia quando necessário. Os documentos em `docs/` e `tcc/README.md` não têm destino obrigatório no Overleaf. Alterações locais não chegam automaticamente ao Overleaf.
