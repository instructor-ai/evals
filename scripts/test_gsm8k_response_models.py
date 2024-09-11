from braintrust import Eval
from autoevals.value import ExactMatch
from datasets import load_dataset
from openai import AsyncOpenAI
from pydantic import BaseModel
import instructor
from asyncio import run
from uuid import uuid4
import json

dataset = load_dataset("567-labs/gsm8k")
oai = AsyncOpenAI()


class Answer(BaseModel):
    chain_of_thought: str
    answer: int


class AnswerWithCalculations(BaseModel):
    chain_of_thought: str
    required_calculations: list[str]
    answer: int


class AssumptionBasedAnswer(BaseModel):
    assumptions: list[str]
    logic_flow: str
    answer: int


class ErrorAwareCalculation(BaseModel):
    key_steps: list[str]
    potential_pitfalls: list[str]
    intermediate_results: list[str]
    answer: int


mode = instructor.Mode.JSON
client = instructor.from_openai(oai, mode=mode)

response_models = [
    Answer,
    AnswerWithCalculations,
    AssumptionBasedAnswer,
    ErrorAwareCalculation,
]


async def main():
    uuid = uuid4()
    print(f"Running eval with uuid: {uuid}")

    full_dataset = list(load_dataset("567-labs/gsm8k", split="test"))
    dataset = full_dataset[:200]
    for response_model in response_models:

        async def task(question, hooks):
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that can solve math problems. Answer the question with the correct response",
                    },
                    {"role": "user", "content": question},
                ],
                response_model=response_model,
            )

            hooks.meta(
                response_model_name=response_model.__name__,
                response_model=json.dumps(response_model.model_json_schema()),
                response=resp.model_dump_json(),
            )
            return resp.answer

        await Eval(
            name="567-labs/gsm8k",
            experiment_name=f"gsm8k-{response_model.__name__}-{uuid}",
            data=lambda: [
                {
                    "input": row["question"],
                    "expected": row["answer"],
                }
                for row in dataset
            ],  # Replace with your eval dataset
            task=task,
            scores=[ExactMatch],
            metadata={
                "model": "gpt-4o-mini",
                "n_samples": len(dataset),
                "response_model": response_model.__name__,
                "mode": mode.value,
            },
            max_concurrency=10,
        )


run(main())
