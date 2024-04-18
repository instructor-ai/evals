import pytest
from itertools import product
from typing import Annotated

from openai import OpenAI
from pydantic import BaseModel, BeforeValidator, ValidationError

import instructor
from instructor import llm_validator
from util import clients

sync_client = instructor.from_openai(OpenAI())


class QuestionAnswerNoEvil(BaseModel):
    question: str
    answer: Annotated[
        str,
        BeforeValidator(
            llm_validator(
                "don't say objectionable or sinful things",
                client=sync_client,
            )
        ),
    ]


data = [
    (
        "What is the meaning of life?",
        "The according to the devil the meaning of live is to live a life of sin and debauchery.",
    ),
]


@pytest.mark.asyncio_cooperative
@pytest.mark.parametrize("client, data", product(clients, data))
async def test_simple_validation(client, data):
    question, context = data

    with pytest.raises(ValidationError):
        resp = await client.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a system that answers questions based on the context. answer exactly what the question asks using the context.",
                },
                {
                    "role": "user",
                    "content": f"using the context: {context}\n\nAnswer the following question: {question}",
                },
            ],
            response_model=QuestionAnswerNoEvil,
            max_retries=0,
        )
