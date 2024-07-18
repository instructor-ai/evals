from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import instructor
from enum import Enum
import os


class Models(str, Enum):
    GPT35TURBO = "gpt-3.5-turbo"
    GPT4TURBO = "gpt-4-turbo"
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
    CLAUDE3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE3_OPUS = "claude-3-opus-20240229"
    CLAUDE3_HAIKU = "claude-3-haiku-20240307"
    CLAUDE35_SONNET = "claude-3-5-sonnet-20240620"


clients = (
    instructor.from_openai(
        AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY")),
        model=Models.GPT4O_MINI,
    ),
    instructor.from_openai(
        AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY")),
        model=Models.GPT4O,
    ),
    instructor.from_openai(
        AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY")),
        model=Models.GPT35TURBO,
    ),
    instructor.from_openai(
        AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY")),
        model=Models.GPT4TURBO,
    ),
    instructor.from_anthropic(
        AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY")),
        model=Models.CLAUDE3_OPUS,
        max_tokens=4000,
    ),
    instructor.from_anthropic(
        AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY")),
        model=Models.CLAUDE3_SONNET,
        max_tokens=4000,
    ),
    instructor.from_anthropic(
        AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY")),
        model=Models.CLAUDE3_HAIKU,
        max_tokens=4000,
    ),
)
