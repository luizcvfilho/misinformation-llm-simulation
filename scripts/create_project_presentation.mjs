import {
  Presentation,
  PresentationFile,
  column,
  fill,
  fixed,
  fr,
  grid,
  grow,
  hug,
  layers,
  panel,
  row,
  rule,
  shape,
  text,
  wrap,
} from "@oai/artifact-tool";
import { mkdir } from "node:fs/promises";
import { join } from "node:path";

const SLIDE_W = 1920;
const SLIDE_H = 1080;
const ROOT = "C:/Users/lcvf1/OneDrive/Documentos/UFRJ/TCC/misinformation-llm-simulation";
const OUT_DIR = join(ROOT, "output", "presentations");
const DECK_PATH = join(OUT_DIR, "misinformation_project_progress.pptx");

const colors = {
  ink: "#172026",
  muted: "#5F6B72",
  paper: "#F7F4EE",
  white: "#FFFFFF",
  line: "#D8D0C2",
  green: "#277260",
  teal: "#1B8992",
  blue: "#356AA0",
  red: "#B55345",
  amber: "#D69B2D",
  plum: "#684D78",
  dark: "#132126",
  softGreen: "#DDEDE5",
  softBlue: "#DCE8F4",
  softAmber: "#F4E8CA",
  softRed: "#F1DAD4",
};

const font = {
  display: "Aptos Display",
  body: "Aptos",
};

const presentation = Presentation.create({
  slideSize: { width: SLIDE_W, height: SLIDE_H },
});

function compose(slide, node) {
  slide.compose(node, {
    frame: { left: 0, top: 0, width: SLIDE_W, height: SLIDE_H },
    baseUnit: 8,
  });
}

function title(textValue, subtitle, options = {}) {
  return column(
    {
      name: options.name ?? "title-stack",
      width: fill,
      height: hug,
      gap: 14,
    },
    [
      text(textValue, {
        name: `${options.name ?? "slide"}-title`,
        width: wrap(options.titleWidth ?? 1380),
        height: hug,
        style: {
          fontFace: font.display,
          fontSize: options.size ?? 58,
          bold: true,
          color: options.color ?? colors.ink,
          lineSpacingMultiple: 0.95,
        },
      }),
      subtitle
        ? text(subtitle, {
            name: `${options.name ?? "slide"}-subtitle`,
            width: wrap(options.subtitleWidth ?? 1180),
            height: hug,
            style: {
              fontFace: font.body,
              fontSize: options.subtitleSize ?? 25,
              color: options.subtitleColor ?? colors.muted,
            },
          })
        : rule({
            name: `${options.name ?? "slide"}-rule`,
            width: fixed(220),
            stroke: options.ruleColor ?? colors.green,
            weight: 5,
          }),
    ].filter(Boolean),
  );
}

function footer(label = "Projeto TCC - Simulação de desinformação com LLMs") {
  return text(label, {
    name: "footer",
    width: fill,
    height: hug,
    style: { fontFace: font.body, fontSize: 15, color: "#7D8589" },
  });
}

function root(children, options = {}) {
  return layers(
    { name: "slide-layers", width: fill, height: fill },
    [
      shape({
        name: "background",
        width: fill,
        height: fill,
        fill: options.bg ?? colors.paper,
        line: { fill: options.bg ?? colors.paper, width: 0 },
      }),
      column(
        {
          name: "slide-root",
          width: fill,
          height: fill,
          padding: options.padding ?? { x: 96, y: 72 },
          gap: options.gap ?? 32,
        },
        children,
      ),
    ],
  );
}

function pill(label, color, width = 220) {
  return panel(
    {
      name: `pill-${label.toLowerCase().replaceAll(" ", "-")}`,
      width: fixed(width),
      height: fixed(42),
      padding: { x: 18, y: 8 },
      fill: color,
      line: { fill: color, width: 0 },
      borderRadius: "rounded-full",
      align: "center",
      justify: "center",
    },
    text(label, {
      width: fill,
      height: hug,
      style: {
        fontFace: font.body,
        fontSize: 18,
        bold: true,
        color: colors.white,
        alignment: "center",
      },
    }),
  );
}

function openMetric(value, label, color) {
  return column(
    { name: `metric-${label}`, width: fill, height: hug, gap: 8 },
    [
      text(value, {
        name: `metric-value-${label}`,
        width: fill,
        height: hug,
        style: {
          fontFace: font.display,
          fontSize: 72,
          bold: true,
          color,
          alignment: "center",
        },
      }),
      text(label, {
        name: `metric-label-${label}`,
        width: fill,
        height: hug,
        style: {
          fontFace: font.body,
          fontSize: 21,
          color: colors.muted,
          alignment: "center",
        },
      }),
    ],
  );
}

function bulletList(items, options = {}) {
  return column(
    { name: options.name ?? "bullet-list", width: fill, height: hug, gap: options.gap ?? 18 },
    items.map((item, index) =>
      row(
        { name: `bullet-row-${index}`, width: fill, height: hug, gap: 16, align: "start" },
        [
          shape({
            name: `bullet-mark-${index}`,
            width: fixed(12),
            height: fixed(12),
            fill: options.color ?? colors.green,
            line: { fill: options.color ?? colors.green, width: 0 },
            borderRadius: "rounded-full",
          }),
          text(item, {
            name: `bullet-text-${index}`,
            width: fill,
            height: hug,
            style: {
              fontFace: font.body,
              fontSize: options.fontSize ?? 27,
              color: options.textColor ?? colors.ink,
            },
          }),
        ],
      ),
    ),
  );
}

function stepBlock(number, label, detail, color) {
  return column(
    { name: `step-${number}`, width: fill, height: hug, gap: 14, align: "center" },
    [
      panel(
        {
          name: `step-number-${number}`,
          width: fixed(72),
          height: fixed(72),
          fill: color,
          line: { fill: color, width: 0 },
          borderRadius: "rounded-full",
          align: "center",
          justify: "center",
        },
        text(String(number), {
          width: fill,
          height: hug,
          style: {
            fontFace: font.display,
            fontSize: 34,
            bold: true,
            color: colors.white,
            alignment: "center",
          },
        }),
      ),
      text(label, {
        name: `step-label-${number}`,
        width: fill,
        height: hug,
        style: {
          fontFace: font.display,
          fontSize: 27,
          bold: true,
          color: colors.ink,
          alignment: "center",
        },
      }),
      text(detail, {
        name: `step-detail-${number}`,
        width: fill,
        height: hug,
        style: {
          fontFace: font.body,
          fontSize: 19,
          color: colors.muted,
          alignment: "center",
        },
      }),
    ],
  );
}

function stdiComponentTable() {
  const rows = [
    ["Componente", "Peso", "Como mede"],
    ["Tema", "0,12", "tópico principal igual ou diferente"],
    ["Subtópicos", "0,12", "distância de Jaccard"],
    ["Entidades", "0,12", "distância de Jaccard"],
    ["Relações", "0,24", "sujeito-ação-objeto"],
    ["Contradição", "0,20", "contradição interna detectada"],
    ["VAD", "0,20", "drift de valência, arousal e dominância"],
  ];
  return column(
    {
      name: "stdi-table",
      width: fill,
      height: hug,
      gap: 0,
    },
    rows.map((cells, index) =>
      row(
        {
          name: `stdi-table-row-${index}`,
          width: fill,
          height: fixed(index === 0 ? 56 : 70),
          gap: 0,
        },
        cells.map((cell, cellIndex) =>
          panel(
            {
              name: `stdi-cell-${index}-${cellIndex}`,
              width: cellIndex === 2 ? grow(1.8) : grow(0.75),
              height: fill,
              padding: { x: 18, y: 12 },
              fill: index === 0 ? colors.green : colors.white,
              line: { fill: colors.line, width: 1 },
              justify: "center",
            },
            text(cell, {
              width: fill,
              height: hug,
              style: {
                fontFace: font.body,
                fontSize: index === 0 ? 19 : 18,
                bold: index === 0 || cellIndex === 0,
                color: index === 0 ? colors.white : colors.ink,
              },
            }),
          ),
        ),
      ),
    ),
  );
}

function horizontalBar(label, valueLabel, ratio, color) {
  return column(
    { name: `bar-${label}`, width: fill, height: hug, gap: 8 },
    [
      row(
        { width: fill, height: hug, align: "center", gap: 16 },
        [
          text(label, {
            width: fixed(240),
            height: hug,
            style: { fontFace: font.body, fontSize: 23, bold: true, color: colors.ink },
          }),
          text(valueLabel, {
            width: fixed(92),
            height: hug,
            style: { fontFace: font.display, fontSize: 28, bold: true, color },
          }),
        ],
      ),
      row(
        { width: fill, height: fixed(28), gap: 0 },
        [
          shape({
            width: grow(Math.max(ratio, 0.02)),
            height: fill,
            fill: color,
            line: { fill: color, width: 0 },
            borderRadius: "rounded-full",
          }),
          shape({
            width: grow(Math.max(1 - ratio, 0.02)),
            height: fill,
            fill: "#E8E1D6",
            line: { fill: "#E8E1D6", width: 0 },
            borderRadius: "rounded-full",
          }),
        ],
      ),
    ],
  );
}

function groupedVadRow(label, trueValue, fakeValue, maxValue = 3.5) {
  return grid(
    {
      name: `vad-row-${label}`,
      width: fill,
      height: hug,
      columns: [fixed(170), fr(1), fixed(74), fr(1), fixed(74)],
      columnGap: 14,
      alignItems: "center",
    },
    [
      text(label, {
        width: fill,
        height: hug,
        style: { fontFace: font.body, fontSize: 22, bold: true, color: colors.ink },
      }),
      row({ width: fill, height: fixed(24), gap: 0 }, [
        shape({ width: grow(trueValue / maxValue), height: fill, fill: colors.green, line: { fill: colors.green, width: 0 }, borderRadius: "rounded-full" }),
        shape({ width: grow(1 - trueValue / maxValue), height: fill, fill: "#E8E1D6", line: { fill: "#E8E1D6", width: 0 }, borderRadius: "rounded-full" }),
      ]),
      text(trueValue.toFixed(3), {
        width: fill,
        height: hug,
        style: { fontFace: font.body, fontSize: 18, color: colors.green, alignment: "center" },
      }),
      row({ width: fill, height: fixed(24), gap: 0 }, [
        shape({ width: grow(fakeValue / maxValue), height: fill, fill: colors.red, line: { fill: colors.red, width: 0 }, borderRadius: "rounded-full" }),
        shape({ width: grow(1 - fakeValue / maxValue), height: fill, fill: "#E8E1D6", line: { fill: "#E8E1D6", width: 0 }, borderRadius: "rounded-full" }),
      ]),
      text(fakeValue.toFixed(3), {
        width: fill,
        height: hug,
        style: { fontFace: font.body, fontSize: 18, color: colors.red, alignment: "center" },
      }),
    ],
  );
}

function vadComparisonChart() {
  return column(
    { name: "vad-comparison-chart", width: fill, height: hug, gap: 24 },
    [
      row(
        { width: fill, height: hug, gap: 24, justify: "end" },
        [pill("True", colors.green, 128), pill("Fake", colors.red, 128)],
      ),
      groupedVadRow("Valência", 2.851, 2.766),
      groupedVadRow("Arousal", 3.148, 3.324),
      groupedVadRow("Dominância", 3.126, 3.157),
      text("Escala VAD prevista pelo modelo; barras normalizadas para leitura comparativa.", {
        width: fill,
        height: hug,
        style: { fontFace: font.body, fontSize: 16, color: colors.muted },
      }),
    ],
  );
}

function makeCover() {
  const slide = presentation.slides.add();
  compose(
    slide,
    layers(
      { name: "cover-layers", width: fill, height: fill },
      [
        shape({
          name: "cover-bg",
          width: fill,
          height: fill,
          fill: colors.dark,
          line: { fill: colors.dark, width: 0 },
        }),
        grid(
          {
            name: "cover-root",
            width: fill,
            height: fill,
            columns: [fr(1.06), fr(0.94)],
            columnGap: 80,
            padding: { x: 112, y: 86 },
          },
          [
            column(
              { name: "cover-title-stack", width: fill, height: fill, gap: 28, justify: "center" },
              [
                text("Da reescrita ao índice de desinformação", {
                  name: "cover-title",
                  width: fill,
                  height: hug,
                  style: {
                    fontFace: font.display,
                    fontSize: 82,
                    bold: true,
                    color: colors.paper,
                    lineSpacingMultiple: 0.92,
                  },
                }),
                rule({ name: "cover-rule", width: fixed(360), stroke: colors.amber, weight: 7 }),
                text(
                  "Relato do desenvolvimento: hipóteses iniciais, auditorias com BERT, STDI, VAD e simulação em grafo.",
                  {
                    name: "cover-subtitle",
                    width: wrap(760),
                    height: hug,
                    style: {
                      fontFace: font.body,
                      fontSize: 28,
                      color: "#C9D4D1",
                    },
                  },
                ),
              ],
            ),
            column(
              { name: "cover-map", width: fill, height: fill, gap: 24, justify: "center" },
              [
                row(
                  { name: "cover-node-row-1", width: fill, height: hug, gap: 22 },
                  [pill("notícia original", colors.green, 250), pill("personalidade", colors.plum, 250)],
                ),
                rule({ name: "cover-path-1", width: fill, stroke: "#6B7C7E", weight: 3 }),
                row(
                  { name: "cover-node-row-2", width: fill, height: hug, gap: 22 },
                  [pill("reescrita", colors.teal, 250), pill("drift semântico", colors.blue, 250)],
                ),
                rule({ name: "cover-path-2", width: fill, stroke: "#6B7C7E", weight: 3 }),
                row(
                  { name: "cover-node-row-3", width: fill, height: hug, gap: 22 },
                  [pill("VAD", colors.amber, 250), pill("STDI", colors.red, 250)],
                ),
              ],
            ),
          ],
        ),
      ],
    ),
  );
}

function makeRoadmap() {
  const slide = presentation.slides.add();
  compose(
    slide,
    root(
      [
        title("A trajetória do projeto foi guiada por uma descoberta central", "Classificar falsidade diretamente não capturava bem a mudança de enquadramento produzida pelos LLMs."),
        shape({
          name: "roadmap-top-spacer",
          width: fill,
          height: fixed(104),
          fill: colors.paper,
          line: { fill: colors.paper, width: 0 },
        }),
        row(
          { name: "roadmap", width: fill, height: hug, gap: 32, align: "start" },
          [
            stepBlock(1, "Reescrever", "Notícias reescritas por modelos e personalidades.", colors.green),
            stepBlock(2, "Auditar", "BERT local e detector pré-treinado como hipóteses de avaliação.", colors.blue),
            stepBlock(3, "Medir drift", "STDI para decompor alterações estruturais do texto.", colors.red),
            stepBlock(4, "Adicionar afeto", "VAD para observar valência, excitação e dominância.", colors.amber),
            stepBlock(5, "Simular rede", "Grafo interativo para cadeias de personalidades.", colors.plum),
          ],
        ),
        shape({
          name: "roadmap-bottom-spacer",
          width: fill,
          height: fixed(118),
          fill: colors.paper,
          line: { fill: colors.paper, width: 0 },
        }),
        footer(),
      ],
    ),
  );
}

function makeInitialPipeline() {
  const slide = presentation.slides.add();
  compose(
    slide,
    root(
      [
        title("Primeiro experimento: notícias reescritas com diferentes personalidades", "O projeto começou testando se a mudança de perspectiva poderia ser capturada por comparação semântica entre original e versão reescrita."),
        grid(
          {
            name: "pipeline-grid",
            width: fill,
            height: grow(1),
            columns: [fr(1), fr(1), fr(1)],
            columnGap: 44,
            alignItems: "center",
          },
          [
            panel(
              { name: "pipeline-data", width: fill, height: fixed(500), fill: colors.softGreen, line: { fill: colors.softGreen, width: 0 }, padding: 34 },
              column({ width: fill, height: fill, gap: 18, justify: "center" }, [
                text("Dados", { width: fill, height: hug, style: { fontFace: font.display, fontSize: 38, bold: true, color: colors.green } }),
                text("NewsData.io e bases locais; seleção por coluna textual, título e identificador.", { width: fill, height: hug, style: { fontFace: font.body, fontSize: 25, color: colors.ink } }),
                text("2315 linhas acumuladas no conjunto newsdata_news.", { width: fill, height: hug, style: { fontFace: font.body, fontSize: 21, color: colors.muted } }),
              ]),
            ),
            panel(
              { name: "pipeline-llms", width: fill, height: fixed(500), fill: colors.softBlue, line: { fill: colors.softBlue, width: 0 }, padding: 34 },
              column({ width: fill, height: fill, gap: 18, justify: "center" }, [
                text("Reescrita", { width: fill, height: hug, style: { fontFace: font.display, fontSize: 38, bold: true, color: colors.blue } }),
                text("Gemini, ChatGPT, Grok, OpenRouter e Llama local.", { width: fill, height: hug, style: { fontFace: font.body, fontSize: 25, color: colors.ink } }),
                text("Personalidades: direita conservadora, esquerda progressista, conspiratória e cética investigativa.", { width: fill, height: hug, style: { fontFace: font.body, fontSize: 21, color: colors.muted } }),
              ]),
            ),
            panel(
              { name: "pipeline-audit", width: fill, height: fixed(500), fill: colors.softAmber, line: { fill: colors.softAmber, width: 0 }, padding: 34 },
              column({ width: fill, height: fill, gap: 18, justify: "center" }, [
                text("Auditoria", { width: fill, height: hug, style: { fontFace: font.display, fontSize: 38, bold: true, color: "#9A6A13" } }),
                text("Comparação semântica original vs. reescrita com BERT/NLI.", { width: fill, height: hug, style: { fontFace: font.body, fontSize: 25, color: colors.ink } }),
                text("Objetivo inicial: sinalizar versões inconsistentes ou potencialmente falsas.", { width: fill, height: hug, style: { fontFace: font.body, fontSize: 21, color: colors.muted } }),
              ]),
            ),
          ],
        ),
        footer("Fonte: README, execution_report.md e módulos src/misinformation_simulation."),
      ],
      { gap: 34 },
    ),
  );
}

function makeBertFindings() {
  const slide = presentation.slides.add();
  compose(
    slide,
    root([
      title("Por que falso ou verdadeiro não bastava", "O problema central não era só detectar mentira factual, mas entender quando uma reescrita passa a alterar a interpretação da notícia."),
      grid(
        {
          name: "binary-limit-grid",
          width: fill,
          height: grow(1),
          columns: [fr(1.2), fr(0.8)],
          columnGap: 60,
          alignItems: "center",
        },
        [
          column(
            { name: "continuum-scale", width: fill, height: hug, gap: 28 },
            [
              text("A mudança acontece em escala, não em chave liga/desliga", {
                width: fill,
                height: hug,
                style: { fontFace: font.display, fontSize: 35, bold: true, color: colors.ink },
              }),
              row(
                { name: "continuum-row", width: fill, height: hug, gap: 14, align: "center" },
                [
                  pill("mesma noticia", colors.green, 130),
                  rule({ width: fixed(32), stroke: colors.line, weight: 4 }),
                  pill("nova enfase", colors.teal, 130),
                  rule({ width: fixed(32), stroke: colors.line, weight: 4 }),
                  pill("novo enquadramento", colors.blue, 190),
                  rule({ width: fixed(32), stroke: colors.line, weight: 4 }),
                  pill("distorcao", colors.amber, 130),
                  rule({ width: fixed(32), stroke: colors.line, weight: 4 }),
                  pill("desinformacao", colors.red, 170),
                ],
              ),
              text("A mesma base factual pode ganhar outro foco, outra causalidade percebida e outro tom. A pergunta mais útil passa a ser: quanto a informação mudou a cada transformação?", {
                width: wrap(980),
                height: hug,
                style: { fontFace: font.body, fontSize: 27, color: colors.muted },
              }),
            ],
          ),
          column({ name: "binary-lessons", width: fill, height: hug, gap: 24 }, [
            bulletList(
              [
                "Uma reescrita pode contar a mesma notícia sem inventar novos dados.",
                "Mesmo assim, pode mudar o sentido prático para quem lê.",
                "A falsidade factual é apenas uma parte da desinformação.",
                "Por isso o projeto passou a medir drift informacional ao longo da cadeia de reescritas.",
              ],
              { color: colors.red, fontSize: 25 },
            ),
          ]),
        ],
      ),
      footer("Virada metodológica: sair da classificação binária e medir mudança informacional."),
    ]),
  );
}

function makeSTDI() {
  const slide = presentation.slides.add();
  compose(
    slide,
    root([
      title("STDI: um índice para medir drift estruturado do texto", "O índice decompõe a mudança em dimensões interpretáveis, em vez de tratar toda diferença textual como falso/verdadeiro."),
      grid(
        {
          name: "stdi-grid",
          width: fill,
          height: grow(1),
          columns: [fr(1.05), fr(0.95)],
          columnGap: 56,
          alignItems: "center",
        },
        [
          stdiComponentTable(),
          column({ name: "stdi-formula", width: fill, height: hug, gap: 24 }, [
            text("Representação extraída", {
              width: fill,
              height: hug,
              style: { fontFace: font.display, fontSize: 36, bold: true, color: colors.ink },
            }),
            bulletList(
              [
                "main_topic",
                "subtopics",
                "central_entities",
                "central_relations no formato sujeito, ação, objeto",
                "narrative_frame",
                "has_internal_contradiction",
                "VAD: valência, arousal e dominância",
              ],
              { color: colors.green, fontSize: 24, gap: 14 },
            ),
            text("STDI com VAD = 0,12 tema + 0,12 subtópico + 0,12 entidade + 0,24 relação + 0,20 contradição + 0,20 VAD", {
              name: "stdi-equation",
              width: fill,
              height: hug,
              style: { fontFace: font.body, fontSize: 23, bold: true, color: colors.red },
            }),
          ]),
        ],
      ),
      footer("Implementado em src/misinformation_simulation/topic_drift/metrics.py."),
    ]),
  );
}

function makeSTDIChain() {
  const slide = presentation.slides.add();
  compose(
    slide,
    root([
      title("O STDI também foi preparado para cadeias de reescrita", "Isso permite avaliar não apenas a distância da versão final para o original, mas o acúmulo de drift em cada passo da simulação."),
      grid(
        {
          name: "chain-grid",
          width: fill,
          height: grow(1),
          columns: [fr(1), fr(1), fr(1)],
          columnGap: 48,
          alignItems: "center",
        },
        [
          openMetric("vs original", "cada versão comparada à notícia inicial", colors.green),
          openMetric("incremental", "cada versão comparada ao passo anterior", colors.blue),
          openMetric("cumulativo", "soma do drift ao longo da cadeia", colors.red),
        ],
      ),
      row(
        { name: "chain-flow", width: fill, height: hug, gap: 22, align: "center" },
        [
          pill("original", colors.green, 210),
          rule({ width: fixed(180), stroke: colors.line, weight: 4 }),
          pill("persona A", colors.blue, 210),
          rule({ width: fixed(180), stroke: colors.line, weight: 4 }),
          pill("persona B", colors.plum, 210),
          rule({ width: fixed(180), stroke: colors.line, weight: 4 }),
          pill("persona C", colors.red, 210),
        ],
      ),
      footer("Funções: calculate_stdi_chain_metrics, annotate_stdi_for_rewrites e annotate_stdi_for_version_chain."),
    ]),
  );
}

function makeVAD() {
  const slide = presentation.slides.add();
  compose(
    slide,
    root([
      title("VAD adiciona a camada emocional", "O estudo Fake vs True mostrou diferenças consistentes de valência, arousal e dominância entre notícias falsas e reais."),
      grid(
        {
          name: "vad-grid",
          width: fill,
          height: grow(1),
          columns: [fr(1.15), fr(0.85)],
          columnGap: 60,
          alignItems: "center",
        },
        [
          vadComparisonChart(),
          column({ name: "vad-copy", width: fill, height: hug, gap: 26 }, [
            text("Sinais observados", {
              width: fill,
              height: hug,
              style: { fontFace: font.display, fontSize: 36, bold: true, color: colors.ink },
            }),
            bulletList(
              [
                "Fake: menor valência média, sugerindo tom mais negativo.",
                "Fake: arousal bem maior, indicando linguagem mais intensa/ativadora.",
                "Dominância teve diferença menor, mas estatisticamente visível no conjunto.",
                "No STDI com VAD, o drift emocional entra com peso 0,20.",
              ],
              { color: colors.amber, fontSize: 24 },
            ),
          ]),
        ],
      ),
      footer("Modelo VAD: RobroKools/vad-bert. Amostras: 10.000 fake e 10.000 true."),
    ]),
  );
}

function makeGraphSimulation() {
  const slide = presentation.slides.add();
  compose(
    slide,
    root([
      title("A simulação evoluiu para um grafo de interação entre personalidades", "A reescrita deixou de ser um evento isolado e passou a ser uma cadeia configurável de transformações."),
      grid(
        {
          name: "graph-grid",
          width: fill,
          height: grow(1),
          columns: [fr(1.05), fr(0.95)],
          columnGap: 58,
          alignItems: "center",
        },
        [
          column({ name: "graph-diagram", width: fill, height: hug, gap: 22 }, [
            row({ width: fill, height: hug, gap: 20 }, [pill("notícia", colors.green, 190), rule({ width: fixed(150), stroke: colors.line, weight: 4 }), pill("nó 1", colors.blue, 170)]),
            row({ width: fill, height: hug, gap: 20 }, [pill("saída 1", colors.teal, 190), rule({ width: fixed(150), stroke: colors.line, weight: 4 }), pill("nó 2", colors.plum, 170)]),
            row({ width: fill, height: hug, gap: 20 }, [pill("saída 2", colors.amber, 190), rule({ width: fixed(150), stroke: colors.line, weight: 4 }), pill("STDI", colors.red, 170)]),
          ]),
          bulletList(
            [
              "Cada nó define provedor, modelo e personalidade.",
              "O backend normaliza uma cadeia conectada e executa os passos em ordem topológica.",
              "Cada passo salva texto de origem, texto reescrito, status, erro, estrutura tópica e métricas.",
              "A execução inclui retry, rate limit, fallback de título e persistência em JSON/JSONL.",
            ],
            { color: colors.plum, fontSize: 25 },
          ),
        ],
      ),
      footer("Implementado em src/misinformation_simulation/simulation/graph.py e scripts/run_interaction_graph.py."),
    ]),
  );
}

function makeUI() {
  const slide = presentation.slides.add();
  compose(
    slide,
    root([
      title("A interface visual tornou a simulação explorável", "O Streamlit Studio reduz o atrito para montar grafos, escolher personalidades e inspecionar resultados sem editar JSON manualmente."),
      grid(
        {
          name: "ui-grid",
          width: fill,
          height: grow(1),
          columns: [fr(1), fr(1), fr(1)],
          columnGap: 40,
          alignItems: "center",
        },
        [
          panel(
            { name: "ui-data", width: fill, height: fixed(480), fill: colors.white, line: { fill: colors.line, width: 2 }, padding: 30 },
            column({ width: fill, height: fill, gap: 20 }, [
              text("Dados", { width: fill, height: hug, style: { fontFace: font.display, fontSize: 35, bold: true, color: colors.green } }),
              bulletList(["carregar CSV/JSON/JSONL", "pré-visualizar colunas", "selecionar texto, título e id"], { fontSize: 22, color: colors.green, gap: 12 }),
            ]),
          ),
          panel(
            { name: "ui-editor", width: fill, height: fixed(480), fill: colors.white, line: { fill: colors.line, width: 2 }, padding: 30 },
            column({ width: fill, height: fill, gap: 20 }, [
              text("Grafo", { width: fill, height: hug, style: { fontFace: font.display, fontSize: 35, bold: true, color: colors.blue } }),
              bulletList(["adicionar/remover nós", "reordenar cadeia", "exportar configuração JSON"], { fontSize: 22, color: colors.blue, gap: 12 }),
            ]),
          ),
          panel(
            { name: "ui-results", width: fill, height: fixed(480), fill: colors.white, line: { fill: colors.line, width: 2 }, padding: 30 },
            column({ width: fill, height: fill, gap: 20 }, [
              text("Resultados", { width: fill, height: hug, style: { fontFace: font.display, fontSize: 35, bold: true, color: colors.red } }),
              bulletList(["resumo por nó", "resumo por notícia", "métricas e saídas por etapa"], { fontSize: 22, color: colors.red, gap: 12 }),
            ]),
          ),
        ],
      ),
      footer("Interface: src/misinformation_simulation/apps/interaction_graph_app.py."),
    ]),
  );
}

function makeEngineering() {
  const slide = presentation.slides.add();
  compose(
    slide,
    root([
      title("Também houve amadurecimento de engenharia do projeto", "Além dos experimentos, o trabalho ganhou organização reprodutível e pontos de extensão para novas análises."),
      grid(
        {
          name: "engineering-grid",
          width: fill,
          height: grow(1),
          columns: [fr(1), fr(1)],
          columnGap: 70,
          alignItems: "center",
        },
        [
          bulletList(
            [
              "Pacote Python em src/misinformation_simulation.",
              "Comandos Make para setup, notebooks, lint, coleta de notícias e UI.",
              "Dependências gerenciadas com uv e Python 3.12+.",
              "Ruff, pre-commit, nbstripout, pytest e coverage configurados.",
            ],
            { color: colors.green, fontSize: 26 },
          ),
          bulletList(
            [
              "Notebooks separados por hipótese: reescrita, BERT local, detector pré-treinado, STDI e VAD.",
              "Relatórios de execução em output/runs e output/execution_report.md.",
              "Testes cobrindo datasets, LLM clients, retry/rate limit, VAD, STDI e UI do grafo.",
              "Persistência dos resultados da simulação em summary JSON e steps JSONL.",
            ],
            { color: colors.blue, fontSize: 26 },
          ),
        ],
      ),
      footer("Fontes: README.md, pyproject.toml, Makefile e diretório tests/."),
    ]),
  );
}

function makeNextSteps() {
  const slide = presentation.slides.add();
  compose(
    slide,
    root([
      title("Status e próximos passos", "O projeto agora mede sinais de desinformação como drift interpretável, mas ainda precisa transformar o índice em evidência validada."),
      grid(
        {
          name: "next-grid",
          width: fill,
          height: grow(1),
          columns: [fr(1), fr(1)],
          columnGap: 72,
          alignItems: "center",
        },
        [
          column({ name: "current-state", width: fill, height: hug, gap: 20 }, [
            text("Já realizado", { width: fill, height: hug, style: { fontFace: font.display, fontSize: 40, bold: true, color: colors.green } }),
            bulletList(
              [
                "Pipeline de reescrita com múltiplos provedores.",
                "Avaliações com BERT local e detector pré-treinado.",
                "STDI com componentes temáticos, relacionais, contraditórios e emocionais.",
                "Estudo VAD em base fake/true e interface visual de simulação.",
              ],
              { color: colors.green, fontSize: 25 },
            ),
          ]),
          column({ name: "next-steps", width: fill, height: hug, gap: 20 }, [
            text("A consolidar", { width: fill, height: hug, style: { fontFace: font.display, fontSize: 40, bold: true, color: colors.red } }),
            bulletList(
              [
                "Definir limiares interpretáveis para baixo, médio e alto drift.",
                "Calibrar melhor os pesos do STDI, incluindo o peso relativo do VAD.",
                "Rodar amostras maiores e comparar por modelo/personalidade.",
                "Validar STDI contra anotação humana ou casos sintéticos controlados.",
                "Aprimorar visualizações da UI para evolução temporal e comparação entre cadeias.",
              ],
              { color: colors.red, fontSize: 25 },
            ),
          ]),
        ],
      ),
      footer(),
    ]),
  );
}

async function exportDeck() {
  await mkdir(OUT_DIR, { recursive: true });

  const builders = [
    ["cover", makeCover],
    ["roadmap", makeRoadmap],
    ["initialPipeline", makeInitialPipeline],
    ["bertFindings", makeBertFindings],
    ["stdi", makeSTDI],
    ["stdiChain", makeSTDIChain],
    ["vad", makeVAD],
    ["graphSimulation", makeGraphSimulation],
    ["ui", makeUI],
    ["engineering", makeEngineering],
    ["nextSteps", makeNextSteps],
  ];
  for (const [name, builder] of builders) {
    builder();
  }

  const pptxBlob = await PresentationFile.exportPptx(presentation);
  await pptxBlob.save(DECK_PATH);

  console.log(JSON.stringify({ deckPath: DECK_PATH }, null, 2));
}

await exportDeck();
