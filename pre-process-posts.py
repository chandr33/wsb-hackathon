import spacy
import pandas as pd
import sys
import re
import numpy as np

IGNORE_LIST = [
    "etf",
    "nyse",
    "nasdaq",
    "wsb",
    "wallstreetbets",
    "fund",
    "congress",
    "u.s.",
    "united states",
    "asperger/autism network",
    "dfv",
    "motley fool",
    "sec",
    "robinhood",
    "ish ish asa asa",
    "senate",
    "house",
    "risky finance",
    "spac",
    "brokers",
    "management",
    "reddit",
    "cnbc" "bloomberg",
    "covid",
]

nlp = spacy.load("en_core_web_trf")


def process(doc):
    for ent in doc.ents:
        if any(term in str(ent).lower() for term in IGNORE_LIST):
            return None
        if ent.label_ == "ORG":
            print(ent)
            return ent
    return None


def preprocess_pipe(texts):
    preproc_pipe = []
    for doc in nlp.pipe(texts, batch_size=20):
        ent = process(doc)
        preproc_pipe.append(ent)
    return preproc_pipe


if __name__ == "__main__":

    df_posts = pd.read_csv("data/posts_df.csv", skip_blank_lines=True, index_col=0)
    df_posts["match_ent"] = preprocess_pipe(df_posts["text"])
    df_posts = df_posts.dropna(subset=["match_ent"])
    df_posts["text"] = df_posts["text"].replace(r"\\n", " ", regex=True)
    df_posts.text = df_posts.text.apply(lambda x: f"{x[:4995]}-----" if len(x) > 5000 else x)

    df_posts = df_posts[df_posts["match_ent"].notna()]
    df_posts = df_posts.assign(external_reference_type="P")
    df_posts = df_posts.assign(company_id="")
    df_posts = df_posts.assign(momentum_factor=1.0)
    df_posts = df_posts.assign(cusip="")
    df_posts = df_posts.assign(isin="")
    df_posts = df_posts.assign(ticker="")
    df_posts = df_posts.assign(exchange="")

    df_posts.rename(columns={"post_id": "external_reference_id", "date": "timestamp", "text": "source_text"}, inplace=True)
    filtered_df = df_posts[
        [
            "external_reference_id",
            "external_reference_type",
            "company_id",
            "momentum_factor",
            "cusip",
            "isin",
            "ticker",
            "exchange",
            "timestamp",
            "source_text",
            "match_ent",
        ]
    ]
    print("Processed posts_df")
    print(filtered_df)
    filtered_df.to_csv("processed_posts.csv")
