import os
import json
import pandas as pd
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from utils.processing import preprocess_text


def load_model(repo_id="HuggingFaceH4/zephyr-7b-beta"):
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        max_new_tokens=1024,
        do_sample=False,
        repetition_penalty=1.03,
    )
    model = ChatHuggingFace(llm=llm)
    return model


def load_data(fpath, sheet_name):
    # Laod in the data
    df = pd.read_excel(fpath, sheet_name)

    if sheet_name == "Train":
        # Change column names
        df.rename(
            columns={
                "tweet_text": "tweet",
                "emotion_in_tweet_is_directed_at": "brand_product_name",
                "is_there_an_emotion_directed_at_a_brand_or_product": "emotion_category",
            },
            inplace=True,
        )
        # Get unique samples
        df.dropna(subset=["tweet"], inplace=True)
        df.drop_duplicates(subset=["tweet"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        # Remove samples that have `I can't tell` as their emotion category
        emotion = "I can't tell"
        df.query("emotion_category != @emotion", inplace=True)
        return df

    if sheet_name == "Test":
        # Change column name
        df.rename(columns={"Tweet": "tweet"}, inplace=True)
        # Get unique samples
        df.dropna(subset=["tweet"], inplace=True)
        df.drop_duplicates(subset=["tweet"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df


def get_samples_with_no_brand_product_association(
    df, process_txt=True, fpath="data_to_aug.jsonl"
):
    if os.path.exists(fpath):
        print(f"File '{fpath}' exists. Reading data...")
        with open(fpath, "r") as file:
            return [json.loads(line) for line in file]

    # Get samples in which no brand or product name is assicated with the tweet.
    df_no_brand_product_association = df[df["brand_product_name"].isna()]
    # df_no_brand_product_association.reset_index(drop=True, inplace=True)
    if process_txt:
        df_no_brand_product_association["tweet"] = df_no_brand_product_association[
            "tweet"
        ].apply(preprocess_text)

    # Store them
    with open("data_to_aug.jsonl", "w") as file:
        for row in df_no_brand_product_association.itertuples(index=True):
            idx, tweet = row.Index, row.tweet
            jsonl_element = json.dumps({"id": idx, "tweet": tweet})
            file.write(jsonl_element)
            file.write("\n")

    # Return the data as a list of dictionaries
    return [
        {"id": idx, "tweet": tweet}
        for idx, tweet in zip(
            df_no_brand_product_association.index,
            df_no_brand_product_association["tweet"],
        )
    ]
