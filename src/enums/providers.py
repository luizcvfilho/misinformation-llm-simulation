from enum import StrEnum


class Provider(StrEnum):
    GEMINI = "gemini"
    CHATGPT = "chatgpt"
    OPENROUTER = "openrouter"
    DEEPSEEK = "deepseek"
    GROK = "grok"
    LOCAL = "local"
