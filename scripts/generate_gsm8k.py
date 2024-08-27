from datasets import load_dataset, Dataset, DatasetDict

splits = ["test", "train"]


def generate_gsm8k(split):
    ds = load_dataset("gsm8k", "main", split=split, streaming=True)
    for row in ds:
        reasoning, answer = row["answer"].split("####")
        answer = int(answer.strip().replace(",", ""))
        yield {
            "question": row["question"],
            "answer": answer,
            "reasoning": reasoning,
        }


# Create the dataset for train and test splits
train_dataset = Dataset.from_generator(lambda: generate_gsm8k("train"))
test_dataset = Dataset.from_generator(lambda: generate_gsm8k("test"))

# Combine them into a DatasetDict
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

dataset.push_to_hub("567-labs/gsm8k")
