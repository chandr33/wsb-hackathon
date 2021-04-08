import spacy
import pandas as pd
import sys


def process(raw_comment, nlp):
    doc = nlp(raw_comment)
    for ent in doc.ents:
        if ent.label_ == "ORG":
            print(ent)
            return ent, raw_comment


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_trf")

    df = pd.read_csv("comments.csv")
    df_list = df["body"][:1000].tolist()
    for item in df_list:
        process(item, nlp)
