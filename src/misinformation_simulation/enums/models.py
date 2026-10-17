from enum import StrEnum
from types import MappingProxyType

from .providers import Provider


class Models(StrEnum):
    GEMINI35FlashLite = "gemini-3.5-flash-lite"
    GPT6Luna = "gpt-6-luna"
    GPT56Luna = "gpt-5.6-luna"
    GPT5Mini = "gpt-5-mini"
    GPT5Nano = "gpt-5-nano"
    GPT41Mini = "gpt-4.1-mini"
    NEMOTRON3Super120B = "nvidia/nemotron-3-super-120b-a12b:free"
    LLAMA318B = "llama3.1:8b"
    STEP35FLASH = "stepfun/step-3.5-flash:free"
    GROKBuild01 = "grok-build-0.1"


DEFAULT_LLM_MODEL = Models.GPT6Luna

MODEL_PROVIDERS = MappingProxyType(
    {
        Models.GEMINI35FlashLite: Provider.GEMINI,
        Models.GPT6Luna: Provider.CHATGPT,
        Models.GPT56Luna: Provider.CHATGPT,
        Models.GPT5Mini: Provider.CHATGPT,
        Models.GPT5Nano: Provider.CHATGPT,
        Models.GPT41Mini: Provider.CHATGPT,
        Models.NEMOTRON3Super120B: Provider.OPENROUTER,
        Models.LLAMA318B: Provider.LOCAL,
        Models.STEP35FLASH: Provider.OPENROUTER,
        Models.GROKBuild01: Provider.GROK,
    }
)


def models_for_provider(provider: Provider | str) -> list[str]:
    provider = Provider(str(provider).strip().lower())
    return [model.value for model in Models if MODEL_PROVIDERS[model] == provider]


def validate_provider_model(provider: Provider | str, model: str) -> None:
    """Reject catalog mismatches while allowing provider-specific custom model IDs."""
    provider = Provider(str(provider).strip().lower())
    model = model.strip()
    if not model:
        raise ValueError("Select or enter a model for the selected provider.")
    expected = MODEL_PROVIDERS.get(model)
    if expected is not None and provider != expected:
        raise ValueError(f"Model '{model}' belongs to provider '{expected}', not '{provider}'.")


def default_model_for_provider(provider: Provider | str) -> str:
    models = models_for_provider(provider)
    return DEFAULT_LLM_MODEL.value if DEFAULT_LLM_MODEL in models else next(iter(models), "")
