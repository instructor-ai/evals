from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import instructor
from enum import Enum


class Models(str, Enum):
    GPT35TURBO = "gpt-3.5-turbo"
    GPT4TURBO = "gpt-4-turbo"
    CLAUDE3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE3_OPUS = "claude-3-opus-20240229"
    CLAUDE3_HAIKU = "claude-3-haiku-20240307"


clients = (
    instructor.from_openai(
        AsyncOpenAI(),
        model=Models.GPT35TURBO,
    ),
    instructor.from_openai(
        AsyncOpenAI(),
        model=Models.GPT4TURBO,
    ),
    instructor.from_anthropic(
        AsyncAnthropic(),
        model=Models.CLAUDE3_OPUS,
        max_tokens=4000,
    ),
    instructor.from_anthropic(
        AsyncAnthropic(),
        model=Models.CLAUDE3_SONNET,
        max_tokens=4000,
    ),
    instructor.from_anthropic(
        AsyncAnthropic(),
        model=Models.CLAUDE3_HAIKU,
        max_tokens=4000,
    ),
)
