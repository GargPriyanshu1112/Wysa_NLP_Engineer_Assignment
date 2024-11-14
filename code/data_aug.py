import os
import re
import json
import pandas as pd
from langchain_core.prompts import PromptTemplate

from utils import (
    load_model,
    load_data,
    get_samples_with_no_brand_product_association,
    preprocess_text,
    sanitize_output,
)

# Set up paths
excel_filepath = r"D:\Priyanshu\Wysa\dataset.xlsx"

# Prompt template
prompt_template = PromptTemplate.from_template(
    """You are an intelligent AI assistant.

A tweet has been provided to you. Your task is to identify the specific brand or product targeted in the provided tweet. Focus solely on determining the main brand or product being referenced.

If the tweet fits one of the categories below, respond with the category name.

### Brand/Product Categories:
  - iPhone
  - iPad or iPhone App
  - iPad
  - Google
  - Android
  - Apple
  - Android App
  - Other Google product or service
  - Other Apple product or service

### Instructions:
  - Your response must contain *only* one of the following:
    * The name of a single category from the list above, based on the primary brand or product mentioned in the tweet.
    * If the tweet does not relate to any of the above categories, respond with only `None`.
  - If the tweet could potentially belong to multiple categories, choose the single category that is the closest match. Do not elaborate on or explain your reasoning process.
  - Do NOT include any additional information, explanations, or comments in your answer.

### Answer Format:
{{"id": string, "category": string}}

Given tweet:
{tweet}
"""
)


def get_response(model, text):
    # processed_text = preprocess_text(text)
    # prompt = prompt_template.format(tweet=processed_text)
    prompt = prompt_template.format(tweet=text)
    model_output = model.invoke(prompt).content
    data_dict = json.loads(model_output)
    id_ = int(data_dict.get("id"))
    category = sanitize_output(data_dict.get("category", ""))
    return json.dumps({"id": id_, "category": category})


def get_batched_response(model, texts, save_csv_path="aug_data.csv"):
    # For cases when batch contains only 1 sample
    if not isinstance(texts, list):
        texts = [texts]

    prompts = []
    for text in texts:
        # processed_text = preprocess_text(text)
        # prompt = prompt_template.format(tweet=processed_text)
        prompt = prompt_template.format(tweet=text)
        prompts.append(prompt)
    model_output_list = model.batch(prompts)

    # Check if the file exists
    file_exists = save_csv_path and os.path.exists(save_csv_path)

    ids, categories = [], []
    for output in model_output_list:
        match = re.search(r"\{(.*?)\}", output.content)
        if not match:
            print("Invalid model output format.")
            continue
        output_content = match.group(0)

        try:
            data_dict = json.loads(output_content)
            ids.append(int(data_dict.get("id")))
            categories.append(sanitize_output(data_dict.get("category", "None")))
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e} for content: {output_content}")
            continue
        except TypeError as e:
            print(f"TypeError: {e} for content: {output_content}")
            continue

    data_dict = {"id": ids, "category": categories}
    df = pd.DataFrame(data_dict)
    df.to_csv(save_csv_path, mode="a", index=False, header=not file_exists)


if __name__ == "__main__":
    # Load the model
    model = load_model(repo_id="HuggingFaceH4/zephyr-7b-beta")
    # Load the training data
    df = load_data(excel_filepath, "Train")
    # Get samples in which no brand or product name is assicated with the tweet
    data_samples_no_brand_product_association = (
        get_samples_with_no_brand_product_association(
            df, process_txt=True, fpath="data_to_aug.jsonl"
        )
    )
    # Get augmented data
    get_batched_response(
        model, data_samples_no_brand_product_association, save_csv_path="aug_data.csv"
    )
