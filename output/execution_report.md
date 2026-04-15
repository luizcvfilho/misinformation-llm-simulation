# Execution Report

Auto-generated execution metadata per notebook run.

## 2026-04-14 23:36:28 - Dataset query and profile
- notebook: llm_simulation_workbench.ipynb
- run_id: manual

### Details
### Dataset selected
- newsdata_news

### Dataset total rows loaded
- 2315

### Dataset rows planned per model
- 5

### Dataset column count
- 21

### Dataset columns
- article_id
- title
- link
- description
- content
- full_description
- pubDate
- pubDateTZ
- image_url
- video_url
- source_id
- source_name
- source_priority
- source_url
- source_icon
- language
- country
- category
- creator
- keywords
- duplicate

### Dataset value summary
- **Language**:
  - english
- **Country**:
  - algeria; qatar; palestine; tunisia; egypt; iraq; syria; united arab emirates; libya; jordan; kuwait; saudi arabia; lebanon; bahrain; morocco
  - argentina
  - armenia
  - australia
  - australia; canada; india; united kingdom; france; germany; united states of america; china
  - australia; japan; united states of america; china; canada; united kingdom; france; germany
  - austria; singapore; switzerland; united kingdom; united arab emirates; france; germany; united states of america
  - azerbaijan
  - bahamas
  - bangladesh
- **Category**:
  - breaking
  - breaking; sports
  - breaking; top
  - business
  - business; breaking
  - business; health
  - business; lifestyle
  - business; lifestyle; top
  - business; technology
  - business; technology; top
- **Keywords**:
  - #bermudianchildren; #bermudacricket
  - #bowling
  - #mlb; baseball; mlb; san francisco giants
  - #promotions
  - #vince vaughn; late night tv; vince vaughn; theo von
  - $49.99; machine:; sealer; vacuum; 90kpa; zachvo
  - -; 8.3; form; schroder; trust; &; real; limited; londonmetric; plc; property; estate; investment
  - /platform/nbc; /platform/peacock; /platform/peacock premium; sports; /platform/directv; baseball
  - 18779304; president-donald-trump; u.s.-&-world; athletes; olympics-2028; olympics; transgender
  - 2026 club car championship purse; korn ferry tour purse this week; korn ferry tour prize money this week; 2026 club car championship winner’s share; 2026 club car championship prize money payout; korn ferry tour purse; 2026 club car championship; korn ferry tour prize money
- **Source name**:
  - 101greatgoals.com
  - 123telugu.com
  - 12news
  - 24 News Hd
  - 24/7 Wall St
  - 263chat
  - 411mania
  - 8days
  - 95.5 Wsb
  - 960 The Ref

### Newsdata query details
- **Updated at utc**: 2026-03-27T02:31:22.989471+00:00
- **Total requests**: 8
- **Latest query summary**:
  - **Fetched at utc**: 2026-03-27T02:31:22.945253+00:00
  - **Query**: 
  - **Language**: en
  - **Country**: 
  - **Category**: 
  - **Max records**: 400
  - **Rows fetched**: 400
  - **Rows appended**: 400
  - **Rows skipped as duplicates**: 0
  - **Dataset rows**: 1916-2315 (count: 400)
  - **Csv lines**: 1918-2317 (count: 400)
  - **Result summary**:
    - **Rows fetched**: 400
    - **Date range min**: 2026-03-26 14:29:29
    - **Date range max**: 2026-03-26 14:31:00
    - **Unique sources**: 290
    - **Unique keywords**: 1177
- **Query history**:
  | # | Fetched at utc | Query | Language | Country | Category | Max records | Rows fetched | Rows appended | Rows skipped | Dataset rows | Csv lines |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | 1 | 2026-03-22T21:52:55.096796+00:00 |  | en |  |  | 10 | 10 | 10 | 0 | 1-10 (count: 10) | 3-12 (count: 10) |
  | 2 | 2026-03-22T21:53:05.533432+00:00 |  | en |  |  | 5 | 5 | 5 | 0 | 11-15 (count: 5) | 13-17 (count: 5) |
  | 3 | 2026-03-26T03:24:09.978613+00:00 |  | en |  |  | 300 | 300 | 300 | 0 | 16-315 (count: 300) | 18-317 (count: 300) |
  | 4 | 2026-03-26T03:55:26.775738+00:00 |  | en |  |  | 400 | 400 | 400 | 0 | 316-715 (count: 400) | 318-717 (count: 400) |
  | 5 | 2026-03-27T01:52:06.429035+00:00 |  | en |  |  | 400 | 400 | 400 | 0 | 716-1115 (count: 400) | 718-1117 (count: 400) |
  | 6 | 2026-03-27T02:16:17.261132+00:00 |  | en |  |  | 600 | 600 | 600 | 0 | 1116-1715 (count: 600) | 1118-1717 (count: 600) |
  | 7 | 2026-03-27T02:24:34.114759+00:00 |  | en |  |  | 200 | 200 | 200 | 0 | 1716-1915 (count: 200) | 1718-1917 (count: 200) |
  | 8 | 2026-03-27T02:31:22.945253+00:00 |  | en |  |  | 400 | 400 | 400 | 0 | 1916-2315 (count: 400) | 1918-2317 (count: 400) |
- **Accumulated dataset summary**:
  - **Rows fetched**: 2315
  - **Date range min**: 2026-03-22 09:51:50
  - **Date range max**: 2026-03-26 14:31:00
  - **Unique sources**: 1112
  - **Unique countries**: 138
  - **Unique categories**: 18
  - **Unique keywords**: 6018

---
## 2026-04-14 23:36:28 - LLM rewriting execution
- notebook: llm_simulation_workbench.ipynb
- run_id: manual

### Details
### Llm models used
- gemini::gemini-3.1-flash-lite-preview
- openrouter::nvidia/nemotron-3-super-120b-a12b:free
- openrouter::stepfun/step-3.5-flash:free
- chatgpt::gpt-4.1-mini
- grok::grok-4-1-fast-non-reasoning
- local::llama3.1:8b

### Personality used
- You are a conspiratorial, denialist persona that rejects official explanations. Interpret events as coordinated manipulation by hidden elites and institutions. Prioritize suspicion, hidden motives, and narrative inversion when interpreting and rewriting the news.

### Prompt template used
- Rewrite the news article from the perspective below.

Personality:
{personality}

Required output language: {target_language_name} ({target_language_code}).

Rules:
- Write strictly in {target_language_name} ({target_language_code}).
- Do not output any other language.
- Keep factual content unchanged.
- Do not invent data, numbers, quotes, events, or characters.
- Change only framing, emphasis, tone, vocabulary, and narrative focus according to the personality.
- Do not explain the personality, bias, or reasoning process.
- Return only the rewritten text.

Title:
{title}

Original text:
{original_text}

### Rewrite metrics
| Dataset | Output name | Provider | Model | Duration seconds | Rows requested | Rows success | Rows error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| newsdata_news | gemini_rewritten_df | gemini | gemini-3.1-flash-lite-preview | 47.639 | 5 | 5 | 0 |
| newsdata_news | openrouter_nvidia_rewritten_df | openrouter | nvidia/nemotron-3-super-120b-a12b:free | 76.32 | 5 | 5 | 0 |
| newsdata_news | openrouter_step_rewritten_df | openrouter | stepfun/step-3.5-flash:free | 0.898 | 5 | 0 | 5 |
| newsdata_news | chatgpt_rewritten_df | chatgpt | gpt-4.1-mini | 11.636 | 5 | 5 | 0 |
| newsdata_news | grok_rewritten_df | grok | grok-4-1-fast-non-reasoning | 7.373 | 5 | 5 | 0 |
| newsdata_news | local_llama_rewritten_df | local | llama3.1:8b | 300.437 | 5 | 5 | 0 |

### Total rewritten success
- 25

---
## 2026-04-14 23:38:11 - Pretrained fake-news detector execution
- notebook: pretrained_fake_news_detector_workbench.ipynb
- run_id: manual

### Details
### Rows audited
- 25

### Detector execution seconds
- 17.196

### Pretrained auditor used
- jy46604790/Fake-News-Bert-Detect

### Total dataset rows
- 25

### Audit model metrics
| Audit model | Fake count | Fake rate |
| --- | --- | --- |
| jy46604790/Fake-News-Bert-Detect | 0 | 0.0 |

### Detector dataset metrics
| Dataset name | Source file | Rows audited | Fake count | Fake rate |
| --- | --- | --- | --- | --- |
| chatgpt_rewritten_df | chatgpt_rewritten_df.csv | 5 | 0 | 0.0 |
| gemini_rewritten_df | gemini_rewritten_df.csv | 5 | 0 | 0.0 |
| grok_rewritten_df | grok_rewritten_df.csv | 5 | 0 | 0.0 |
| local_llama_rewritten_df | local_llama_rewritten_df.csv | 5 | 0 | 0.0 |
| openrouter_nvidia_rewritten_df | openrouter_nvidia_rewritten_df.csv | 5 | 0 | 0.0 |

### Execution details
- **Dataset selector**: ALL
- **Input dir**: ..\output\rewritten
- **Rewritten column**: rewritten_news

---
## 2026-04-14 23:40:39 - Local consistency audit execution
- notebook: bert_fake_real_workbench.ipynb
- run_id: manual

### Details
### Rows audited
- 25

### Audit execution seconds
- 44.623

### Local trained auditor used
- MoritzLaurer/mDeBERTa-v3-base-mnli-xnli

### Total dataset rows
- 25

### Audit model metrics
| Audit model | Fake count | Fake rate |
| --- | --- | --- |
| MoritzLaurer/mDeBERTa-v3-base-mnli-xnli | 1 | 0.04 |

### Audit dataset metrics
| Dataset name | Source file | Rows audited | Fake count | Fake rate |
| --- | --- | --- | --- | --- |
| local_llama_rewritten_df | local_llama_rewritten_df.csv | 5 | 1 | 0.2 |
| chatgpt_rewritten_df | chatgpt_rewritten_df.csv | 5 | 0 | 0.0 |
| gemini_rewritten_df | gemini_rewritten_df.csv | 5 | 0 | 0.0 |
| grok_rewritten_df | grok_rewritten_df.csv | 5 | 0 | 0.0 |
| openrouter_nvidia_rewritten_df | openrouter_nvidia_rewritten_df.csv | 5 | 0 | 0.0 |

### Execution details
- **Dataset selector**: ALL
- **Input dir**: ..\output\rewritten
- **Original column**: description
- **Rewritten column**: rewritten_news

---
