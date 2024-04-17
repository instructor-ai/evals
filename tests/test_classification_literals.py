from itertools import product
from typing import Literal
from util import clients
from pydantic import BaseModel

import pytest


class ClassifySpam(BaseModel):
    label: Literal["spam", "not_spam"]


data = [
    ("I am a spammer who sends many emails every day", "spam"),
    ("I am a responsible person who does not spam", "not_spam"),
]


@pytest.mark.asyncio_cooperative
@pytest.mark.parametrize("client, data", product(clients, data))
async def test_classification(client, data):
    input, expected = data
    prediction = await client.create(
        response_model=ClassifySpam,
        messages=[
            {
                "role": "system",
                "content": "Classify this text as 'spam' or 'not_spam'.",
            },
            {
                "role": "user",
                "content": input,
            },
        ],
    )
    assert prediction.label == expected
