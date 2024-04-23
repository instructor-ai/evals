import pytest
from itertools import product
from pydantic import AfterValidator, BaseModel, Field
from typing import Annotated
from util import clients
from langsmith import unit


def uppercase_validator(v):
    if v.islower():
        raise ValueError("Name must be ALL CAPS")
    return v


class UserDetail(BaseModel):
    name: Annotated[str, AfterValidator(uppercase_validator)] = Field(
        ..., description="The name of the user"
    )
    age: int


data = [
    (
        "Extract `jason is 12`",
        ("JASON", 12),
    ),
    (
        "Extract `danny is 125 years old`",
        ("DANNY", 125),
    ),
    (
        "Extract `DONALD is a 45 year old man`",
        ("DONALD", 45),
    ),
]


@pytest.mark.asyncio_cooperative
@pytest.mark.parametrize("client, data", product(clients, data))
async def test_retries(client, data):
    await check_retries(client, data)

@unit
async def check_retries(client, data):
    query, expected = data
    response = await client.create(
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": query},
        ],
        max_retries=3,
    )
    assert response.name == expected[0]
    assert response.age == expected[1]
