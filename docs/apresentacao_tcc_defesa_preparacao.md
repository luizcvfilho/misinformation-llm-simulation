# Preparação da apresentação final do TCC

**Apresentação final:** `output/presentations/tcc_deriva_informacional_defesa_final.pptx`  
**PDF final:** `output/presentations/tcc_deriva_informacional_defesa_final.pdf`  

Esta versão preserva os ajustes manuais realizados no PPTX v4 em 25/09/2026 e foi exportada diretamente pelo PowerPoint. As versões intermediárias da mesma apresentação foram removidas.

## Estrutura

A apresentação contém **29 slides principais** e **5 slides de apoio**. A meta é concluir os slides principais em 25–30 minutos. Os slides de apoio devem ser usados somente durante a arguição.

| Bloco | Slides | Tempo sugerido |
| --- | ---: | ---: |
| Abertura, motivação e pergunta | 1–5 | 5 min |
| Auditorias, literatura e construção do STDI | 6–15 | 9 min |
| Corpus e método experimental | 16–20 | 5 min |
| Resultados | 21–25 | 7 min |
| Conclusão e referências | 26–29 | 3 min |
| Apoio para arguição | 30–34 | fora do tempo principal |

## Slides metodológicos acrescentados

| Slide | Função na narrativa |
| ---: | --- |
| 9 | Explicar a reescrita: entrada, prompt, perspectiva, restrições e encadeamento entre versões. |
| 10 | Mostrar como o LLM converte cada texto em tema, domínio, subtópicos, entidades, relações, enquadramento e contradição interna. |
| 11 | Apresentar os 17 domínios primários de `topic_domain`, escolhidos pelo LLM no vocabulário controlado IPTC Media Topics. |
| 12 | Registrar a evolução do Jaccard após a extração estruturada para a comparação semântica vigente, incluindo a salvaguarda por domínio temático. |
| 33 | Apresentar o experimento de TF-IDF: objetivo, representação, particionamento e comparação de desempenho com os atributos estruturais do STDI. |
| 34 | Mostrar exemplos dos n-gramas de maior coeficiente positivo e negativo no modelo TF-IDF. |

Esses quatro slides formam o elo entre o limite da classificação binária e a definição do STDI. No slide 11, o LLM seleciona um domínio temático primário no vocabulário controlado IPTC Media Topics; `null` é usado quando o domínio não pode ser determinado.

## Conteúdo ainda a preencher

Os blocos amarelos com a marca **A PREENCHER** indicam somente informações ainda indisponíveis ou análises não concluídas.

| Slide | Pendência | Critério de conclusão |
| ---: | --- | --- |
| 1 | Banca e data da defesa | Usar os dados registrados na documentação oficial da defesa. |
| 16 | Protocolo do corpus | Registrar hash, regra de seleção, período, fontes e licenças. |
| 31 | Versões dos modelos | Conferir nomes, revisões, parâmetros, prompts e datas de execução. |

## Recomendações para a versão entregue à banca

1. **Alinhar pergunta, hipóteses e resultados.** Cada hipótese deve ter um resultado correspondente. Se o desenho permanecer exploratório, usar perguntas analíticas em vez de hipóteses direcionais.
2. **Completar a fundamentação.** O slide de trabalhos relacionados deve mostrar como cada linha de literatura sustenta uma parte do método e qual lacuna o STDI ocupa.
3. **Fixar o corpus.** Registrar arquivo, hash, data, critérios e amostra usada. A forte concentração em notícias de negócios precisa aparecer como limitação.
4. **Mostrar um exemplo concreto.** A banca precisa enxergar como tema, entidades, relações, contradição e VAD se alteram em um texto real.
5. **Relatar incerteza.** Para contrastes entre cenários, incluir tamanho de efeito e intervalos, preservando o pareamento por notícia.
6. **Distinguir resultado de interpretação.** STDI mede deriva textual; não demonstra falsidade, persuasão, crença ou propagação social.
7. **Fechar com uma resposta direta.** A conclusão deve dizer o que foi observado sobre tipo de perspectiva, ordem e domínio temático, além da contribuição metodológica.
8. **Preparar a arguição.** Manter nos slides de apoio as versões de modelos, a tabela de cenários, sensibilidade e exemplos de falhas.
9. **Ensaiar com cronômetro.** Buscar 25–28 minutos para conservar margem para transições e eventuais interrupções.

## Critério para remover placeholders

Antes de exportar a versão entregue à banca, buscar por `A PREENCHER`, `[AUTOR, ano]`, `[nome completo]` e `[data]`. Nenhum desses marcadores deve permanecer na versão final.
