from braintrust import Eval, Score
from autoevals.value import ExactMatch
from datasets import load_dataset
from openai import AsyncOpenAI
from pydantic import BaseModel
import instructor
from asyncio import run
from uuid import uuid4

dataset = load_dataset("567-labs/gsm8k")
oai = AsyncOpenAI()


class Answer(BaseModel):
    chain_of_thought: str
    answer: int


modes = [instructor.Mode.TOOLS, instructor.Mode.TOOLS_STRICT]


async def main():
    uuid = uuid4()
    print(f"Running eval with uuid: {uuid}")
    for eval_mode in modes:
        client = instructor.from_openai(oai, mode=eval_mode)
        dataset = list(load_dataset("567-labs/gsm8k", split="test"))

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
                response_model=Answer,
            )
            hooks.meta(
                reasoning=resp.chain_of_thought,
            )
            return resp.answer

        await Eval(
            name="567-labs/gsm8k",
            experiment_name=f"gsm8k-{eval_mode}-{uuid}",
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
                "mode": str(eval_mode),
                "n_samples": len(dataset),
            },
            max_concurrency=10,
        )


run(main())
