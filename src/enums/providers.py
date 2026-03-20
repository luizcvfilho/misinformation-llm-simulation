from enum import Enum


class Provider(str, Enum):
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    DEEPSEEK = "deepseek"
    LOCAL = "local"
