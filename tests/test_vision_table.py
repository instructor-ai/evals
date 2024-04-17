from io import StringIO
from typing import Annotated, Any, List
from pydantic import (
    BaseModel,
    BeforeValidator,
    PlainSerializer,
    InstanceOf,
    WithJsonSchema,
)
import pandas as pd
import pytest
from itertools import product
from util import clients
from instructor import AsyncInstructor


def md_to_df(data: Any) -> Any:
    if isinstance(data, str):
        return (
            pd.read_csv(
                StringIO(data),  # Get rid of whitespaces
                sep="|",
                index_col=1,
            )
            .dropna(axis=1, how="all")
            .iloc[1:]
            .map(lambda x: x.strip())
        )  # type: ignore
    return data


MarkdownDataFrame = Annotated[
    InstanceOf[pd.DataFrame],
    BeforeValidator(md_to_df),
    PlainSerializer(lambda x: x.to_markdown()),
    WithJsonSchema(
        {
            "type": "string",
            "description": """
                The markdown representation of the table, 
                each one should be tidy, do not try to join tables
                that should be seperate""",
        }
    ),
]


class Table(BaseModel):
    caption: str
    dataframe: MarkdownDataFrame


class MultipleTables(BaseModel):
    tables: List[Table]


urls = [
    "https://a.storyblok.com/f/47007/2400x1260/f816b031cb/uk-ireland-in-three-charts_chart_a.png/m/2880x0",
    "https://a.storyblok.com/f/47007/2400x2000/bf383abc3c/231031_uk-ireland-in-three-charts_table_v01_b.png/m/2880x0",
]


@pytest.mark.asyncio_cooperative
@pytest.mark.parametrize("client, url", product(clients, urls))
async def test_extract(client: AsyncInstructor, url: str):
    if client.kwargs["model"] != "gpt-4-turbo":
        pytest.skip("Only OpenAI supported for now, we need to support images for both")

    resp = await client.create(
        response_model=MultipleTables,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
                            First, analyze the image to determine the most appropriate headers for the tables.
                            Generate a descriptive h1 for the overall image, followed by a brief summary of the data it contains. 
                            For each identified table, create an informative h2 title and a concise description of its contents.
                            Finally, output the markdown representation of each table.


                            Make sure to escape the markdown table properly, and make sure to include the caption and the dataframe.
                            including escaping all the newlines and quotes. Only return a markdown table in dataframe, nothing else.
                        """,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    },
                ],
            }
        ],
    )
    assert len(resp.tables) > 0
