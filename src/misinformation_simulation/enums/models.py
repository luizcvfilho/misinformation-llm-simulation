from enum import StrEnum


class Models(StrEnum):
    GEMINI31FlashLite = "gemini-3.1-flash-lite-preview"
    GPT56Luna = "gpt-5.6-luna"
    GPT5Mini = "gpt-5-mini"
    GPT5Nano = "gpt-5-nano"
    GPT41Mini = "gpt-4.1-mini"
    NEMOTRON3Super120B = "nvidia/nemotron-3-super-120b-a12b:free"
    LLAMA318B = "llama3.1:8b"
    STEP35FLASH = "stepfun/step-3.5-flash:free"
    GROK41FastNonReasoning = "grok-4-1-fast-non-reasoning"


DEFAULT_LLM_MODEL = Models.GPT56Luna
