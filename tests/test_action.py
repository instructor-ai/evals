import instructor
from openai import OpenAI
from typing import Iterable, List, Optional
from enum import Enum
from pydantic import BaseModel, ValidationError
import pytest
from itertools import product
from util import clients


class PriorityEnum(str, Enum):
    high = "High"
    medium = "Medium"
    low = "Low"


class Subtask(BaseModel):
    """Correctly resolved subtask from the given transcript"""

    id: int
    name: str


class Ticket(BaseModel):
    """Correctly resolved ticket from the given transcript"""

    id: int
    name: str
    description: str
    priority: PriorityEnum
    assignees: List[str]
    subtasks: Optional[List[Subtask]]
    dependencies: Optional[List[int]]


# Apply the patch to the OpenAI client
# enables response_model keyword
client = instructor.from_openai(OpenAI())


def generate(data: str) -> Iterable[Ticket]:
    return client.chat.completions.create(
        model="gpt-4",
        response_model=Iterable[Ticket],
        messages=[
            {
                "role": "system",
                "content": "The following is a transcript of a meeting...",
            },
            {
                "role": "user",
                "content": f"Create the action items for the following transcript: {data}",
            },
        ],
    )


EXAMPLE = """
Alice: Hey team, we have several critical tasks we need to tackle for the upcoming release. First, we need to work on improving the authentication system. It's a top priority.

Bob: Got it, Alice. I can take the lead on the authentication improvements. Are there any specific areas you want me to focus on?

Alice: Good question, Bob. We need both a front-end revamp and back-end optimization. So basically, two sub-tasks.

Carol: I can help with the front-end part of the authentication system.

Bob: Great, Carol. I'll handle the back-end optimization then.

Alice: Perfect. Now, after the authentication system is improved, we have to integrate it with our new billing system. That's a medium priority task.

Carol: Is the new billing system already in place?

Alice: No, it's actually another task. So it's a dependency for the integration task. Bob, can you also handle the billing system?

Bob: Sure, but I'll need to complete the back-end optimization of the authentication system first, so it's dependent on that.

Alice: Understood. Lastly, we also need to update our user documentation to reflect all these changes. It's a low-priority task but still important.

Carol: I can take that on once the front-end changes for the authentication system are done. So, it would be dependent on that.

Alice: Sounds like a plan. Let's get these tasks modeled out and get started."""


@pytest.mark.asyncio_cooperative
@pytest.mark.parametrize("client, data", product(clients, [EXAMPLE]))
async def test_generate_action_items(client, data):
    with pytest.raises(ValidationError):
        await client.create(
            messages=[
                {
                    "role": "system",
                    "content": "The following is a transcript of a meeting...",
                },
                {
                    "role": "user",
                    "content": f"Create the action items for the following transcript: {data}",
                },
            ],
            response_model=Iterable[Ticket],
            max_retries=0,
        )
