# Instructor Evaluations

We use various instructor clients to evaluate the quality and capabilities of extractions and reasoning.

We'll run these tests and see what ends up failing often.

```
pip install -r requirements.txt
pytest 
```

## Adding a New Test

To contribute a new test similar to `test_classification_literals.py`, follow these steps:

1. **Create a New Test File**:
   - Create a new test file in the `tests` directory. For example, `tests/test_new_feature.py`.

2. **Import Necessary Modules**:
   - Import the required modules at the beginning of your test file. You will typically need `pytest`, `product` from `itertools`, `Literal` from `typing`, and `clients` from `util`.

   ```python
   from itertools import product
   from typing import Literal
   from util import clients
   from pydantic import BaseModel
   import pytest
   ```

3. **Define Your Data Model**:
   - Define a Pydantic data model for the expected response. For example, if you are testing a sentiment analysis feature, you might define:

   ```python
   class SentimentAnalysis(BaseModel):
       label: Literal["positive", "negative", "neutral"]
   ```

4. **Prepare Your Test Data**:
   - Prepare a list of tuples containing the input data and the expected output. For example:

   ```python
   data = [
       ("I love this product!", "positive"),
       ("This is the worst experience ever.", "negative"),
       ("It's okay, not great but not bad either.", "neutral"),
   ]
   ```

5. **Write the Test Function**:
   - Write an asynchronous test function using `pytest.mark.asyncio_cooperative` and `pytest.mark.parametrize` to iterate over the clients and data. For example:

   ```python
   @pytest.mark.asyncio_cooperative
   @pytest.mark.parametrize("client, data", product(clients, data))
   async def test_sentiment_analysis(client, data):
       input, expected = data
       prediction = await client.create(
           response_model=SentimentAnalysis,
           messages=[
               {
                   "role": "system",
                   "content": "Analyze the sentiment of this text as 'positive', 'negative', or 'neutral'.",
               },
               {
                   "role": "user",
                   "content": input,
               },
           ],
       )
       assert prediction.label == expected
   ```

6. **Run Your Tests**:
   - Run your tests using `pytest` to ensure everything works as expected.

   ```sh
   pytest
   ```

By following these steps, you can easily add new tests to evaluate different features using various instructor clients. Make sure to keep your tests asynchronous and handle any specific requirements for the feature you are testing.


## Contributions

When contributing just make sure everything is as async and we'll handle the rest!


We could use contributions for almost everything from the examples page! https://useinstructor.com/examples/ 
