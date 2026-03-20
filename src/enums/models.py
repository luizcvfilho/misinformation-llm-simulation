from enum import Enum


class Models(str, Enum):
    GEMINI31FlashLite = "gemini-3.1-flash-lite-preview"
    NEMOTRON3Super120B = "nvidia/nemotron-3-super-120b-a12b:free"
    LLAMA318B = "llama3.1:8b"
    STEP35FLASH = "stepfun/step-3.5-flash:free"