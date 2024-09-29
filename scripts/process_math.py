import json
import os
from datasets import load_dataset
import boto3
from tqdm import tqdm
import io


def extract_relevant_files(parent_path: str):
    metadata = {}

    for root, _, files in os.walk(parent_path):
        for file in files:
            if file == "solution.json":
                solution_path = os.path.abspath(os.path.join(root, "solution.json"))
                with open(solution_path, "r") as f:
                    solutions = json.load(f)

                for solution in solutions:
                    metadata[solution["id"]] = {
                        "file_name": os.path.abspath(
                            os.path.join(
                                os.path.dirname(solution_path), solution["id"] + ".png"
                            )
                        ),
                        "objects": {
                            "data": json.dumps(solution["data"]),
                            "type": solution["type"],
                            "id": solution["id"],
                        },
                    }

    return metadata


def upload_images_to_r2(bucket: str, folder_name: str):
    for root, _, files in os.walk(folder_name):
        for file in tqdm(files, desc="Uploading images to R2"):
            if file.endswith(".png"):
                file_path = os.path.join(root, file)
                with open(file_path, "rb") as file_content:
                    s3.upload_fileobj(io.BytesIO(file_content.read()), bucket, file)


if __name__ == "__main__":
    bucket_name = "math-evals"
    folder_name = "./scripts/data"

    s3 = boto3.client(
        service_name="s3",
        endpoint_url=f"https://{os.environ['CLOUDFLARE_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["CLOUDFLARE_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["CLOUDFLARE_SECRET_ACCESS_KEY_ID"],
        region_name="auto",
    )
    # upload_images_to_r2("math-evals", folder_name)
    metadata = extract_relevant_files(folder_name)
    metadata_file_path = os.path.join(folder_name, "metadata.jsonl")
    with open(metadata_file_path, "w") as metadata_file:
        for entry in metadata.values():
            metadata_file.write(json.dumps(entry) + "\n")

    dataset = load_dataset("imagefolder", data_dir=folder_name)
    dataset.push_to_hub("567-labs/psle-math")
