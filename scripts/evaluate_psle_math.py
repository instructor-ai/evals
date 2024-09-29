from datasets import load_dataset
from asyncio import run
from braintrust import Eval
import instructor
from autoevals.value import ExactMatch
from typing import Union, Literal
import google.generativeai as genai
from pydantic import BaseModel, ConfigDict
import json
from PIL.PngImagePlugin import PngImageFile
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
import base64
import httpx


class Fraction(BaseModel):
    whole: int
    numerator: int
    denominator: int


class MultipleChoice(BaseModel):
    choice: Literal[1, 2, 3, 4]


class Number(BaseModel):
    value: float


class EvaluationItem(BaseModel):
    input: PngImageFile
    expected: Union[Fraction, MultipleChoice, Number]
    metadata: dict[str, str]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MultipleChoiceResponse(BaseModel):
    chain_of_thought: str
    answer: MultipleChoice


class Response(BaseModel):
    chain_of_thought: str
    answer: Union[Fraction, Number]


def format_dataset_braintrust(dataset):
    for row in dataset:
        row = {
            **row["objects"],
            "image": row["image"],
            "data": json.loads(row["objects"]["data"]),
        }
        if row["type"] == "multiple_choice":
            expected_output = MultipleChoice(choice=row["data"]["choice"])
        elif row["type"] == "number":
            expected_output = Number(value=row["data"]["value"])
        elif row["type"] == "fraction":
            expected_output = Fraction(
                whole=row["data"]["whole"],
                numerator=row["data"]["numerator"],
                denominator=row["data"]["denominator"],
            )
        yield EvaluationItem(
            input=row["image"],
            expected=expected_output,
            metadata={"id": row["id"], "type": row["type"]},
        )


def generate_questions():
    for row in format_dataset_braintrust(load_dataset("567-labs/psle-math")["train"]):
        yield row


def get_client(provider: Literal["gemini", "openai", "anthropic"], model: str = ""):
    if provider == "gemini":
        return instructor.from_gemini(
            genai.GenerativeModel(model_name=model),
            mode=instructor.Mode.GEMINI_JSON,
            use_async=True,
        )
    elif provider == "openai":
        return instructor.from_openai(AsyncOpenAI())
    elif provider == "anthropic":
        return instructor.from_anthropic(AsyncAnthropic())


def get_response_model(type: Literal["multiple_choice", "number", "fraction"]):
    if type == "multiple_choice":
        return MultipleChoiceResponse
    else:
        return Response


async def generate_gemini_response(client, input, hooks):
    response_model = get_response_model(input["type"])

    resp = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Solve the following question. Make sure to think through your answer step by step before you provide the final answer",
            },
            {
                "role": "user",
                "content": input["image"],
            },
        ],
        response_model=response_model,
    )
    hooks.meta(
        chain_of_thought=resp.chain_of_thought,
    )
    return resp.answer


async def generate_openai_response(client, input, hooks):
    response_model = get_response_model(input["type"])

    resp = await client.chat.completions.create(
        model=input["model"],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Solve the following question. Make sure to think through your answer step by step before you provide the final answer",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": input["image_url"]["url"],
                        },
                    },
                ],
            }
        ],
        response_model=response_model,
    )
    hooks.meta(
        chain_of_thought=resp.chain_of_thought,
    )
    return resp.answer


async def generate_anthropic_response(client, input, hooks):
    response_model = get_response_model(input["type"])

    resp = await client.messages.create(
        model=input["model"],
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.b64encode(
                                httpx.get(input["image_url"]["url"]).content
                            ).decode("utf-8"),
                        },
                    },
                ],
            },
        ],
        response_model=response_model,
    )
    hooks.meta(
        chain_of_thought=resp.chain_of_thought,
    )
    return resp.answer


async def main():
    provider = "anthropic"
    model = "claude-3-5-sonnet-20240620"

    client = get_client(provider, model)
    dataset = list(generate_questions())

    async def task(input, hooks):
        if provider == "gemini":
            return await generate_gemini_response(client, input, hooks)
        elif provider == "openai":
            return await generate_openai_response(client, input, hooks)
        elif provider == "anthropic":
            return await generate_anthropic_response(client, input, hooks)

    await Eval(
        name="567-labs/psle-math-evals",
        data=[
            {
                "input": {
                    "image_url": {
                        "url": f"https://r2-worker.evals.workers.dev/{row.metadata['id']}.png",
                    },
                    "image": row.input,
                    "type": row.metadata["type"],
                    "model": model,
                },
                "expected": row.expected,
                "metadata": row.metadata,
            }
            for row in dataset
        ],
        task=task,
        scores=[ExactMatch],
        metadata={"model": model, "provider": provider},
        max_concurrency=10,
    )


if __name__ == "__main__":
    run(main())
