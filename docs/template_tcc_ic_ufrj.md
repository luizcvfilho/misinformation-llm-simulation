# Guia do template de TCC do IC/UFRJ

Este documento consolida as regras observadas no template `ModeloTCC-Latex-2025.zip` e define como elas devem orientar a escrita, a revisão e a formatação do TCC neste projeto, inclusive quando a edição for feita no Overleaf.

## 1. Fonte e versão documentada

- Arquivo de origem: `ModeloTCC-Latex-2025.zip`.
- Instituição indicada pelo modelo: Universidade Federal do Rio de Janeiro, Instituto de Computação, Bacharelado em Ciência da Computação.
- Data dos arquivos no ZIP: 10/11/2025.
- Data desta inspeção: 22/09/2026.
- SHA-256 do ZIP inspecionado: `9bceba56a35aedc84c690e95f03f87b4139eb3fa9bb4fccccb71024b75e7ce20`.
- Arquivo principal: `main.tex`.
- Formatação corrente: `formatacao.tex` e `url16023.sty`, localizados na raiz do ZIP.

A pasta `Old/` contém uma implementação anterior e não deve ser usada como fonte de regras, comandos ou estrutura enquanto a versão corrente compilar corretamente.
### 1.1 Fonte canônica no repositório

A estrutura exata recebida foi preservada em:

- arquivo original: `templates/modelo-tcc-ic-ufrj-2025/archive/ModeloTCC-Latex-2025.zip`;
- conteúdo extraído: `templates/modelo-tcc-ic-ufrj-2025/source/`;
- instruções de preservação e uso: `templates/modelo-tcc-ic-ufrj-2025/README.md`.

A árvore original disponível para inspeção é:

```text
source/
├── main.tex
├── formatacao.tex
├── url16023.sty
├── elementos-pretextuais/
│   ├── personalizar.tex
│   ├── folhadeaprovacao.tex
│   ├── dedicatoria.tex
│   ├── agradecimentos.tex
│   ├── epigrafe.tex
│   ├── resumo.tex
│   ├── abstract.tex
│   ├── abreviaturas.tex
│   └── simbolos.tex
├── elementos-textuais/
│   ├── capitulo-1.tex
│   ├── capitulo-2.tex
│   ├── capitulo-3.tex
│   ├── capitulo-4.tex
│   └── capitulo-5.tex
├── elementos-postextuais/
│   ├── referencias.bib
│   ├── glossario.tex
│   ├── apendice-1.tex
│   ├── apendice-2.tex
│   ├── anexo-1.tex
│   └── anexo-2.tex
├── figuras/
├── codigos/
└── Old/
```

Os arquivos de `source/` formam uma cópia de referência e não devem receber o texto do TCC. Ao iniciar ou reconstruir o projeto no Overleaf, copiar toda essa árvore para uma área de trabalho e editar a cópia. Não recriar a estrutura manualmente a partir deste resumo.


## 2. Hierarquia de autoridade

Em caso de divergência, seguir esta ordem:

1. orientação explícita e atual do aluno, do orientador, da banca ou do IC/UFRJ;
2. arquivos correntes do template, fora da pasta `Old/`;
3. este guia consolidado;
4. convenções gerais da ABNT e sugestões de estilo.

Textos em vermelho, referências fictícias, imagens demonstrativas, nomes de pessoas, capítulos de exemplo e trechos em `Lorem ipsum` são material didático do template. Eles devem ser removidos ou substituídos, não incorporados ao TCC.

Se uma regra institucional nova divergir deste guia, atualizar este documento e registrar a origem e a data da mudança.

## 3. Configuração técnica que deve ser preservada

O template corrente usa:

```latex
\documentclass[12pt, a4paper, english, brazil, oneside,
  chapter=TITLE, section=TITLE]{abntex2}
```

- tamanho de fonte base: 12 pt;
- papel: A4;
- idiomas: português do Brasil e inglês;
- classe: `abntex2`;
- compilador compatível com a configuração atual: pdfLaTeX;
- codificação do arquivo: UTF-8;
- codificação de fonte: T1;
- sistema bibliográfico: BibTeX com `abntex2cite`;
- estilo de citações: autor-data, pela opção `alf`;
- títulos destacados nas referências conforme a opção `abnt-emphasize=bf`.

Não migrar o projeto para XeLaTeX, LuaLaTeX, `biblatex` ou Biber sem uma necessidade concreta e uma validação completa do PDF, pois isso pode alterar a formatação institucional.

Os arquivos `formatacao.tex` e `url16023.sty` são marcados pelo próprio modelo como arquivos que não devem ser modificados. Mudanças neles só devem ocorrer para corrigir incompatibilidade confirmada ou cumprir orientação institucional posterior. Toda mudança deve ser pequena, documentada e conferida visualmente.

## 4. Uso no Overleaf

1. Criar um projeto com **New Project > Upload Project** e enviar o ZIP do template.
2. Manter `main.tex` na raiz e defini-lo como **Main document** nas configurações do projeto.
3. Selecionar **pdfLaTeX** como compilador.
4. Manter todos os caminhos relativos do ZIP, especialmente `elementos-pretextuais/`, `elementos-textuais/`, `elementos-postextuais/`, `figuras/` e `codigos/`.
5. Compilar antes de substituir os exemplos, para confirmar que o template original funciona no ambiente escolhido.

O Overleaf documenta que o documento principal deve, preferencialmente, permanecer na raiz do projeto: <https://docs.overleaf.com/getting-started/recompiling-your-project/the-main-document>.

### Sincronização com este repositório

- O arquivo `references.bib` deste repositório é a fonte bibliográfica canônica e verificada.
- No Overleaf, substituir o conteúdo demonstrativo de `elementos-postextuais/referencias.bib` pelo conteúdo do `references.bib` do repositório. Assim, a linha já existente no `main.tex` pode ser preservada:

```latex
\bibliography{elementos-postextuais/referencias.bib}
```

- Quando uma referência for adicionada, corrigida ou removida, atualizar tanto `references.bib` quanto a cópia usada no Overleaf.
- Antes de solicitar alterações diretas nos arquivos LaTeX em conversas futuras, exportar ou sincronizar a versão mais recente do projeto do Overleaf. Não presumir que a cópia local esteja atualizada.
- Não manter duas chaves diferentes para a mesma obra.
### Comunicação obrigatória das alterações do Overleaf

Ao concluir qualquer tarefa de escrita, revisão ou formatação, a resposta deve conter uma seção chamada **Alterações para replicar no Overleaf**. Essa seção deve:

- listar todos os arquivos alterados ou que precisam de alteração, inclusive `main.tex`;
- informar o caminho exato de cada arquivo no projeto do Overleaf;
- descrever objetivamente o que deve ser adicionado, substituído, removido ou enviado;
- fornecer o conteúdo pronto para copiar ou identificar com precisão o trecho afetado;
- avisar quando a mudança exigir recompilação completa, atualização de referências ou envio de uma figura;
- nunca pressupor que alterações feitas neste repositório chegaram automaticamente ao Overleaf.

Para a bibliografia, informar explicitamente o mapeamento:

```text
Repositório: references.bib
Overleaf:    elementos-postextuais/referencias.bib
```

Se a tarefa exigir somente uma alteração em `main.tex`, declarar: “Nenhum arquivo além de `main.tex` precisa ser alterado no Overleaf.” Se não houver alteração em arquivo algum, declarar que não há sincronização necessária.


## 5. Responsabilidade dos arquivos

| Caminho | Função | Regra de edição |
| --- | --- | --- |
| `main.tex` | Ordem e inclusão dos elementos do trabalho | Alterar apenas para incluir, retirar ou reorganizar elementos e capítulos. |
| `formatacao.tex` | Pacotes e regras visuais | Preservar, salvo correção justificada. |
| `url16023.sty` | Formatação de URLs nas referências | Não modificar. |
| `elementos-pretextuais/personalizar.tex` | Instituição, título, autores, orientação, local, ano e preâmbulo | Substituir todos os dados fictícios. |
| `elementos-pretextuais/*.tex` | Elementos anteriores ao texto | Manter apenas os elementos aplicáveis e substituir os exemplos. |
| `elementos-textuais/capitulo-*.tex` | Corpo científico do TCC | Substituir integralmente o conteúdo demonstrativo. |
| `elementos-postextuais/referencias.bib` | Base usada pelo BibTeX no Overleaf | Sincronizar com `references.bib`. |
| `elementos-postextuais/apendice-*.tex` | Materiais elaborados pelo autor | Manter apenas quando necessários. |
| `elementos-postextuais/anexo-*.tex` | Materiais produzidos por terceiros | Manter apenas quando necessários. |
| `elementos-postextuais/glossario.tex` | Definições e siglas do glossário | Usar somente se o glossário for mantido. |
| `figuras/` | Imagens e gráficos | Substituir as figuras de demonstração. |
| `codigos/` | Listagens de código | Usar somente para trechos necessários ao argumento do TCC. |

## 6. Formatação de página e texto

### Documento digital

O template corrente usa margens de 25 mm em todos os lados:

```latex
\usepackage[margin=25 mm]{geometry}
```

### Versão impressa

O próprio arquivo `formatacao.tex` sugere, para impressão:

```latex
\usepackage[left=30mm, top=30mm, right=20mm, bottom=20mm]{geometry}
```

Também há uma indicação para trocar `oneside` por `doubleside` na versão impressa. Essas duas alterações devem ser feitas apenas quando a forma de entrega final for confirmada com o orientador ou o IC/UFRJ.

### Corpo do texto

- espaçamento de 1,5 linha;
- recuo de 1,25 cm na primeira linha do parágrafo;
- nenhum espaço adicional entre parágrafos;
- separação de 1,5 linha entre títulos de seção e o texto anterior ou posterior;
- espaçamento simples em citações longas, notas, resumo, abstract, referências, legendas e partes indicadas da capa e da folha de rosto;
- início dos elementos textuais com a Introdução;
- cabeçalho dos elementos textuais contendo apenas o número da página, conforme `\pagestyle{simple}`.

### Títulos

O template configura:

- capítulos em negrito e maiúsculas, tamanho normal;
- seções em maiúsculas e sem negrito, tamanho normal;
- subseções em negrito, tamanho normal;
- numeração hierárquica produzida automaticamente pelo LaTeX.

Usar `\chapter`, `\section` e `\subsection`; não simular títulos manualmente com tamanho, negrito ou numeração digitada.

### Paginação

- a contagem começa na folha de rosto;
- os números só aparecem a partir da Introdução;
- a numeração exibida nos elementos textuais usa algarismos arábicos;
- a paginação deve ser gerada pelo template, nunca digitada manualmente.

## 7. Estrutura do trabalho

### Elementos obrigatórios indicados pelo template

1. capa;
2. folha de rosto;
3. ficha catalográfica;
4. folha de aprovação;
5. resumo em português;
6. abstract em inglês;
7. sumário;
8. elementos textuais;
9. referências.

A ficha catalográfica definitiva e a folha de aprovação assinada são normalmente incorporadas na etapa final. Durante a redação, seus marcadores podem permanecer, desde que o PDF de entrega final não contenha imagens ou dados fictícios.

### Elementos opcionais indicados pelo template

- dedicatória;
- agradecimentos;
- epígrafe;
- listas de ilustrações, códigos, tabelas e quadros;
- lista de abreviaturas e siglas;
- lista de símbolos;
- glossário;
- apêndices;
- anexos.

Elementos opcionais sem conteúdo devem ser removidos ou comentados em `main.tex`; não deixar páginas demonstrativas vazias ou fictícias.

### Estrutura textual recomendada para este projeto

O template inclui cinco capítulos apenas como exemplo. A estrutura científica vigente do TCC é:

1. Introdução;
2. Fundamentação teórica e trabalhos relacionados;
3. Evolução metodológica do projeto;
4. Materiais e métodos;
5. Resultados;
6. Discussão;
7. Conclusão.

Essa divisão é uma recomendação do plano de escrita do projeto, não uma exigência formal do template. Ela pode ser ajustada com o orientador. Cada novo capítulo deve ter seu próprio arquivo e ser incluído em `main.tex` com `\input{...}`.

## 8. Resumo, abstract e palavras-chave

O resumo e o abstract devem:

- ser escritos em um único parágrafo;
- usar espaçamento simples;
- conter entre 150 e 500 palavras;
- preferir terceira pessoa do singular e voz ativa;
- evitar símbolos e contrações que não sejam de uso corrente;
- apresentar objetivo, método, resultados e conclusões;
- não conter resultados ainda não consolidados.

As palavras-chave devem:

- aparecer logo abaixo do resumo ou abstract;
- ser introduzidas por `Palavras-chave:` ou `Keywords:`;
- ser separadas por ponto e vírgula;
- terminar com ponto;
- começar com letra minúscula, salvo nomes próprios e nomes científicos.

O abstract deve ser uma tradução conceitualmente fiel da versão final do resumo, sem acrescentar ou omitir conclusões.

## 9. Citações e referências

O template usa citações autor-data com `abntex2cite` e BibTeX.

### Citação indireta

- não usa aspas;
- normalmente não exige página;
- deve ser redigida com palavras próprias, sem alterar o sentido da fonte;
- deve usar uma chave existente na bibliografia, por exemplo `\cite{mohamed2025broken}` ou `\citeonline{mohamed2025broken}`.

### Citação direta curta

- tem até três linhas;
- permanece no parágrafo;
- usa aspas duplas;
- informa autor, ano e página;
- aspas existentes no original passam a aspas simples dentro da citação.

### Citação direta longa

- tem mais de três linhas;
- usa o ambiente `citacao`;
- tem recuo de 4 cm da margem esquerda;
- usa fonte menor, espaçamento simples e nenhuma aspa;
- informa autor, ano e página.

### Citação de citação

O template demonstra o uso de `apud`, mas ele deve ser excepcional. Sempre que possível, consultar e citar a fonte original. Quando isso não for possível, deixar claro qual obra não foi consultada e usar os comandos oferecidos por `abntex2cite` de modo consistente.

### Integridade bibliográfica

- nunca inventar uma referência ou completar metadados por suposição;
- conferir se a fonte sustenta a afirmação específica;
- diferenciar paráfrase, citação direta e interpretação do autor;
- não reutilizar as referências fictícias que acompanham o template;
- não mudar para `biblatex` apenas porque ele é sugerido em tutoriais gerais: este template depende de `abntex2cite` e BibTeX;
- toda chave citada deve existir em `references.bib`;
- toda fonte nova usada no texto deve ser adicionada também a `docs/base_bibliografica_tcc.md`, com sua função e cuidados de interpretação.

## 10. Tabelas, quadros, figuras, equações e códigos

### Tabelas e quadros

- usar tabela para dados quantitativos, normalmente com tratamento numérico ou estatístico;
- usar quadro para informações qualitativas, classificações, comparações e sínteses;
- colocar título com `\caption{...}`;
- atribuir identificador único com `\label{tab:...}` ou `\label{quad:...}`;
- indicar a fonte abaixo com `\legend{Fonte: ...}`;
- em resultados próprios, usar uma formulação como `Fonte: elaboração própria.`;
- não inserir tabelas apenas como imagem quando os dados puderem ser representados em LaTeX.

### Figuras e gráficos

- armazenar os arquivos em `figuras/`;
- usar `\caption`, `\includegraphics`, `\legend` e `\label`;
- informar a origem ou indicar elaboração própria;
- garantir legibilidade no tamanho final da página;
- não distorcer proporções para preencher espaço.

### Remissões

Usar `\label` e `\ref` para capítulos, seções, tabelas, quadros, figuras, equações e códigos. Não digitar números manualmente, pois eles podem mudar durante a redação. No texto, o template recomenda iniciar o tipo de elemento com maiúscula: `Figura`, `Tabela`, `Quadro`, `Seção`, `Capítulo` e `Equação`.

### Código-fonte

Listagens extensas devem ser evitadas no corpo. Inserir somente trechos necessários para explicar o método e mover material complementar para apêndices ou para o repositório. O template oferece `\includecode`, mas o conteúdo científico tem prioridade sobre demonstrações de implementação.

## 11. Alíneas e subalíneas

Quando uma seção sem título precisar enumerar assuntos:

- introduzir a lista com dois-pontos;
- ordenar alíneas por letras minúsculas seguidas de parêntese;
- iniciar o texto das alíneas com letra minúscula;
- encerrar alíneas com ponto e vírgula, exceto a última, que termina com ponto;
- se houver subalíneas, encerrar a alínea introdutória com dois-pontos;
- iniciar subalíneas com travessão e alinhá-las ao texto correspondente.

Preferir os ambientes `alineas` e `subalineas` fornecidos por `abntex2`.

## 12. Elementos pré-textuais opcionais

- **Dedicatória:** breve, na metade inferior da página, alinhada à direita e sem o título “DEDICATÓRIA”.
- **Agradecimentos:** limitar às contribuições relevantes; quando o trabalho tiver financiamento, agradecer às instituições financiadoras.
- **Epígrafe:** relacionada ao tema, alinhada à direita a partir da metade da página e acompanhada da autoria.
- **Siglas:** incluir apenas siglas efetivamente usadas e manter a mesma forma no texto.
- **Símbolos:** incluir apenas símbolos necessários e defini-los de maneira inequívoca.

## 13. Elementos pós-textuais

- referências são obrigatórias;
- glossário é opcional;
- apêndices contêm materiais elaborados pelo próprio autor;
- anexos contêm materiais produzidos por terceiros;
- apêndices e anexos aparecem depois das referências;
- apêndices vêm antes dos anexos;
- identificar cada item por letra maiúscula e título explicativo.

## 14. Regras científicas específicas deste TCC

Além da formatação institucional, toda redação deve seguir `docs/plano_escrita_tcc.md`, `docs/base_bibliografica_tcc.md` e `references.bib`.

Em particular:

- o objeto central é a deriva informacional e afetiva em cadeias de reescrita por agentes baseados em LLMs;
- o STDI operacionaliza dimensões de mudança textual, mas não verifica fatos contra o mundo externo;
- não confundir deriva informacional, contradição interna, veracidade factual, crença, exposição e compartilhamento;
- não apresentar agentes LLM como equivalentes a seres humanos;
- não atribuir causalidade a personas quando o desenho sustentar apenas associação;
- identificar pilotos como exploratórios;
- documentar corpus, amostra, modelo, versão, prompt, parâmetros, data e unidade de análise;
- citar resultados produzidos pelo projeto como resultados deste estudo, não como achados da literatura externa;
- não afirmar que pesos ou faixas do STDI estão validados antes da validação humana prevista.

## 15. Checklist para cada entrega

### Conteúdo

- [ ] A seção cumpre uma função definida no plano de escrita.
- [ ] Afirmações externas têm fontes verificadas e pertinentes.
- [ ] Resultados próprios informam amostra, unidade de análise, cenário e versão relevante.
- [ ] Limitações e interpretações alternativas estão explícitas.
- [ ] Deriva não foi tratada como sinônimo de falsidade.

### Template

- [ ] `main.tex` é o documento principal e compila com pdfLaTeX.
- [ ] Nenhum texto vermelho, nome fictício, figura demonstrativa ou `Lorem ipsum` permaneceu.
- [ ] Elementos opcionais vazios foram removidos ou comentados.
- [ ] Títulos e numeração são gerados pelos comandos estruturais do LaTeX.
- [ ] Figuras, tabelas e quadros têm título, fonte, rótulo e remissão no texto.
- [ ] Não há referências ou remissões indefinidas no log de compilação.
- [ ] Resumo e abstract têm entre 150 e 500 palavras e correspondem entre si.
- [ ] A bibliografia do Overleaf está sincronizada com `references.bib`.
- [ ] Ficha catalográfica e folha de aprovação estão adequadas à etapa da entrega.
- [ ] A configuração para versão digital ou impressa foi confirmada.

## 16. Decisões que ainda exigem confirmação

Antes da versão final, confirmar com o orientador ou o IC/UFRJ:

1. se a entrega será somente digital ou também impressa;
2. se a versão final deve usar `oneside` ou `doubleside`;
3. se as margens finais serão uniformes de 25 mm ou 30/30/20/20 mm;
4. o procedimento vigente para gerar e inserir a ficha catalográfica;
5. a composição e a forma definitiva da folha de aprovação;
6. se evolução metodológica será um capítulo próprio ou parte de Materiais e métodos;
7. quais listas opcionais serão mantidas;
8. se o glossário será necessário;
9. o título final, as palavras-chave e os dados completos de orientação e banca.

