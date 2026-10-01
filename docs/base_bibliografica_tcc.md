# Base bibliográfica comentada do TCC

**Última verificação dos metadados:** 22/09/2026

**Arquivo de citações:** [`references.bib`](../references.bib)

**Plano de uso na escrita:** [`plano_escrita_tcc.md`](plano_escrita_tcc.md)

## 1. Escopo da revisão

Esta base apoia o TCC sobre **deriva informacional e afetiva em cadeias de reescrita de notícias por agentes baseados em LLMs**. A pergunta de pesquisa provisória é:

> Em que medida a ordem e o tipo de perspectiva atribuída a agentes LLM alteram o conteúdo e o enquadramento emocional de uma notícia ao longo de uma cadeia de reescritas?

O trabalho propõe o Structured Topic Drift Index (STDI) para medir alterações de tema, subtópicos, entidades, relações, contradição interna e VAD. O STDI não verifica fatos contra fontes externas e não mede crença, exposição ou compartilhamento humano.

### Classificação usada

- **Núcleo:** deve aparecer no texto, pois fundamenta diretamente o problema, o método ou a validação.
- **Complementar:** útil para comparação, discussão ou decisão metodológica específica.
- **Fora do núcleo:** artigo válido, mas pouco alinhado ao recorte atual; manter apenas se o escopo mudar.

## 2. Conjunto mínimo recomendado

| Chave BibTeX | Referência | Função no TCC | Onde citar | Prioridade |
| --- | --- | --- | --- | --- |
| `liu2025stepwise` | Liu et al. (2025), *The Stepwise Deception* | Trabalho mais próximo: simula a transformação gradual de notícias verdadeiras em falsas com agentes e propõe avaliação multidimensional. Deve ser comparado cuidadosamente ao STDI, destacando diferenças de agentes, cadeia e alvo de medição. | Introdução, trabalhos relacionados, método e discussão | Núcleo |
| `mohamed2025broken` | Mohamed et al. (2025), *LLM as a Broken Telephone* | Evidência direta de que geração iterativa pode acumular distorções. Sustenta o desenho em cadeia e a medição incremental e contra o original. | Introdução, fundamentação e discussão | Núcleo |
| `chuang2024opinion` | Chuang et al. (2024), *Simulating Opinion Dynamics with Networks of LLM-based Agents* | Fundamenta redes de agentes LLM e mostra que vieses induzidos por prompt podem alterar a dinâmica observada. Também alerta para tendências próprias do modelo. | Fundamentação, método e limitações | Núcleo |
| `liu2024skepticism` | Liu et al. (2024), *From Skepticism to Acceptance* | Simulação de propagação de notícias falsas com personas, memória e reflexão. É importante como comparação entre mudança de atitude e mudança do texto. | Trabalhos relacionados e discussão | Núcleo |
| `frisch2024interaction` | Frisch e Giulianelli (2024), *LLM Agents in Interaction* | Sustenta a necessidade de avaliar consistência das personas e alinhamento linguístico durante interações. | Personas, ameaças à validade e discussão | Núcleo |
| `chen2025rpa` | Chen et al. (2025), survey sobre avaliação de role-playing agents | Oferece taxonomia e diretrizes para avaliar agentes com papéis. Ajuda a justificar critérios de fidelidade, consistência e validade. | Fundamentação metodológica e protocolo de avaliação | Núcleo |
| `abdurahman2025primer` | Abdurahman et al. (2025), primer de avaliação de LLMs em ciências sociais | Fundamenta documentação de modelo e prompt, repetição, análise de sensibilidade e validação contra julgamento humano. | Método, validade e limitações | Núcleo |
| `su2024adapting` | Su, Cardie e Nakov (2024), adaptação de detectores de fake news à era dos LLMs | Mostra que autoria humana/máquina e geração/paráfrase afetam detectores. Ajuda a contextualizar por que classificação binária e sinais de estilo são insuficientes para o problema do TCC. | Introdução, detectores iniciais e discussão | Núcleo |
| `yang2025rethink` | Yang et al. (2025), revisão de detecção de rumores na era dos LLMs | Revisão recente para organizar detecção, cognição, interação e comportamento, incluindo modelagem baseada em agentes. | Estado da arte | Núcleo |
| `guo2021truth` | Guo et al. (2021), evolução empírica de verdade para notícia falsa | Antecedente direto da noção de evolução de conteúdo; apresenta o conjunto FNE com estágios verdade → fake → fake evoluída. | Fundamentação da evolução textual | Núcleo |
| `zhang2013rumor` | Zhang et al. (2013), evolução de rumores em redes sociais | Antecedente clássico que trata explicitamente a modificação do conteúdo durante a propagação, em vez de considerar uma mensagem estática. | Fundamentação histórica | Núcleo |
| `entman1993framing` | Entman (1993), teoria de enquadramento | Base conceitual para diferenciar mudança de enquadramento, seleção e saliência de falsidade factual. | Fundamentação conceitual do STDI | Núcleo |
| `lazer2018science` | Lazer et al. (2018), *The Science of Fake News* | Referência interdisciplinar para definir o problema de notícias falsas e situar respostas científicas. | Introdução e definições | Núcleo |
| `vosoughi2018spread` | Vosoughi, Roy e Aral (2018), propagação de notícias verdadeiras e falsas | Evidência empírica clássica sobre diferenças de difusão no Twitter. Não deve ser apresentada como evidência de transformação textual. | Introdução | Núcleo |
| `reimers2019sbert` | Reimers e Gurevych (2019), Sentence-BERT | Fundamenta embeddings de sentenças e comparação por similaridade cosseno. | Método do STDI | Núcleo |
| `wang2020minilm` | Wang et al. (2020), MiniLM | Fundamenta a família do encoder usada por `all-MiniLM-L6-v2`. O artigo não descreve sozinho o ajuste específico do sentence-transformer; citar também SBERT e documentar o identificador do modelo. | Método do STDI | Núcleo |
| `buechel2017emobank` | Buechel e Hahn (2017), EmoBank | Fundamenta a representação dimensional de emoção em valência, arousal e dominância em texto. | Fundamentação e método VAD | Núcleo |
| `mohammad2018vad` | Mohammad (2018), NRC VAD Lexicon | Referência metodológica para VAD e confiabilidade de anotações humanas. | Fundamentação e limitações do VAD | Núcleo |

## 3. Trabalhos complementares

| Chave BibTeX | Uso recomendado | Cuidado de interpretação |
| --- | --- | --- |
| `liu2025mosaic` | Comparar simulação com grafo dirigido, personas, disseminação e moderação. | O foco inclui ações como curtir, compartilhar e sinalizar; o TCC mede transformação textual, não comportamento real de plataforma. |
| `wang2025echo` | Discutir polarização, estruturas de rede e comparação com modelos clássicos de opinião. | Não usar para afirmar que as cadeias atuais reproduzem câmaras de eco; o desenho do TCC não implementa a mesma dinâmica. |
| `mohammad2025vadv2` | Fundamentar o NRC VAD v2 como alternativa lexical na auditoria de contraste contextual. Metadados e método conferidos em 01/10/2026 no artigo e recurso oficial. | Avaliações humanas de termos em inglês; a confiabilidade lexical não valida a agregação por notícia nem o uso em português. Não é um modelo neural de texto completo. |
| `mohammad2025breaking` | Fundamentar as expressões compostas incluídas no NRC VAD v2. Metadados e descrição do recurso conferidos em 01/10/2026 na ACL Anthology. | O casamento por expressão mais longa e a média por ocorrência são decisões desta implementação, não métodos de avaliação textual validados pelo artigo. |
| `wu2024sheepdog` | Sustentar a vulnerabilidade de detectores baseados em estilo a reenquadramentos produzidos por LLMs. | O problema é robustez de detecção, não mensuração de deriva. |
| `whitehouse2022knowledge` | Explicar que conhecimento externo pode melhorar detecção quando a base é pertinente e atualizada. | Reforça a distinção entre STDI e checagem factual; não é um método adotado atualmente. |
| `grootendorst2022bertopic` | Fundamentar uma análise exploratória de temas no corpus ou nas trajetórias de reescrita. | BERTopic não deve substituir os componentes pareados do STDI sem novo experimento e validação. |
| `karnam2026bowling` | Precedente de procedimento: embeddings de rótulos livres com `all-MiniLM-L6-v2`, BERTopic e consolidação qualitativa de tópicos. | O objeto de estudo é interação humano–ChatGPT, não desinformação. Serve como referência de método, não de fenômeno. |

## 4. Trabalhos mantidos fora do núcleo

| Chave BibTeX | Decisão atual | Quando recuperar |
| --- | --- | --- |
| `irnawan2025crosslingual` | **Não priorizar.** Trata verificação multilíngue explicável com evidências externas, distante da medida de transformação em cadeia. | Se o TCC passar a avaliar factualidade externa ou transferência entre idiomas. |
| `zhang2025gas3` | **Não priorizar.** O principal ganho é escala por agentes de grupo e previsão de tráfego de rede. | Se escalabilidade populacional ou dinâmica macroscópica de rede se tornar contribuição do trabalho. |
| `hu2024multimodal` | **Remover do referencial central.** O corpus e o STDI atuais são textuais. | Se imagens, vídeos ou detecção precoce multimodal forem incorporados. |

Não é necessário apagar esses registros do `.bib`: mantê-los documentados evita refazer a triagem caso o escopo mude.

## 5. Mapa de leitura por tema

### Evolução e reescrita da informação

1. `zhang2013rumor`
2. `guo2021truth`
3. `mohamed2025broken`
4. `liu2025stepwise`

### Simulação com agentes, personas e redes

1. `chuang2024opinion`
2. `frisch2024interaction`
3. `liu2024skepticism`
4. `chen2025rpa`
5. `liu2025mosaic`
6. `wang2025echo`

### Desinformação e limites de detectores

1. `lazer2018science`
2. `vosoughi2018spread`
3. `su2024adapting`
4. `wu2024sheepdog`
5. `whitehouse2022knowledge`
6. `yang2025rethink`

### Fundamentos do STDI

1. `entman1993framing`
2. `reimers2019sbert`
3. `wang2020minilm`
4. `buechel2017emobank`
5. `mohammad2018vad`

### Validação científica

1. `abdurahman2025primer`
2. `chen2025rpa`
3. `frisch2024interaction`

## 6. Regras para uso durante a escrita

1. Ler a seção relevante do artigo antes de associá-lo a uma afirmação específica.
2. Evitar citações decorativas: cada citação deve ter uma função identificável.
3. Diferenciar claramente resultados da literatura, resultados deste projeto e hipóteses interpretativas.
4. Não chamar o STDI de detector de fake news, métrica de verdade ou medida de influência social.
5. Ao comparar métodos, relatar diferenças de corpus, idioma, modelo, prompt, unidade de análise e objetivo.
6. Para artigos de 2025–2026, conferir novamente os metadados na versão final do TCC.
7. Se uma nova fonte for usada no texto, adicioná-la simultaneamente a este documento e a `references.bib`.

## 7. Lacunas ainda abertas

- Validar o STDI com julgamento humano e, se possível, um segundo anotador.
- Procurar literatura adicional sobre avaliação de métricas de mudança semântica pareada.
- Definir uma referência específica para o modelo VAD efetivamente usado (`RobroKools/vad-bert`), além das referências gerais de VAD.
- Documentar a origem e a licença do corpus final de notícias.
- Verificar se a banca espera exclusivamente artigos revisados por pares; `grootendorst2022bertopic` e `guo2021truth` exigem atenção ao tipo de publicação registrado.
