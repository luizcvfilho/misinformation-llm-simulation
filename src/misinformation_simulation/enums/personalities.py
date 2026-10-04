from enum import StrEnum

from misinformation_simulation.config.prompts import PERSONALITY_PROMPTS


class DefaultPersonality(StrEnum):
    ConservativeRight = PERSONALITY_PROMPTS["ConservativeRight"]
    ProgressiveLeft = PERSONALITY_PROMPTS["ProgressiveLeft"]
    ConspiracyDenialist = PERSONALITY_PROMPTS["ConspiracyDenialist"]
    InvestigativeSkeptic = PERSONALITY_PROMPTS["InvestigativeSkeptic"]
    EmotionalAmplifier = PERSONALITY_PROMPTS["EmotionalAmplifier"]
    ConciliatoryCommunicator = PERSONALITY_PROMPTS["ConciliatoryCommunicator"]
