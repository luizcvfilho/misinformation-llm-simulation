from enum import StrEnum


class Provider(StrEnum):
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    DEEPSEEK = "deepseek"
    LOCAL = "local"
