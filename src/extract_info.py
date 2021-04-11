import dnalab
import spacy
import pandas as pd
import os
import glob
import time
import numpy as np
import ast

from collections import defaultdict
from multiprocessing import Process, Pool, cpu_count
from src.scrape_api import PROJECT_PATH, POSTS_PATH, COMMENTS_PATH, POSTS_DATA_PATH, COMMENTS_DATA_PATH

BLACKLIST = ['bot', 'nyse', 'wsb', 'stock', 'mod', 'yolo', 'sec', 'fda', 'it', 'be', 'company', 'hospital']
nlp = spacy.load("en_core_web_sm")


def filter_low_confidence_entities(entities, cutoff=0.1):
    return {key: value for key, value in entities.items() if value > cutoff}


def get_momentum(text):
    doc = nlp.make_doc(text)
    beams = nlp.get_pipe("ner").beam_parse([doc], beam_width=1, beam_density=0.0000)
    entity_scores = defaultdict(float)
    total_score = 0
    for score, ents in nlp.get_pipe("ner").moves.get_beam_parses(beams[0]):
        total_score += score
        for start, end, label in ents:
            entity_scores[(start, end, label)] += score

    normalized_beam_score = {k: v / total_score for k, v in entity_scores.items()}
    confident_entities = filter_low_confidence_entities(normalized_beam_score)

    confident_entity_texts = {}

    for k, v in confident_entities.items():
        if k[2] == 'ORG':
            confident_entity_texts[str(doc[int(k[0]): int(k[1])])] = v * 10

    return confident_entity_texts


def process(raw_comment, nlp):
    doc = nlp(raw_comment)
    tokens = []
    res = None
    for ent in doc.ents:
        if ent.text.lower() not in BLACKLIST and ent.label_ == "ORG":
            tokens.append(str(ent))

    if len(tokens) > 0:
        res = ", ".join(tokens)
    return res


def create_smaller_processing_sets(path):
    all_files = glob.glob(path + "/*.csv")
    all_files.sort()
    n = len(all_files)
    chunk_size = n//(cpu_count()-1)
    file_sets = [all_files[i:i+chunk_size]for i in range(0, n, chunk_size)]
    return file_sets


def create_df_from_list(file_list, column_list, df_type):
    dfs = []
    for filename in file_list:
        df = pd.read_csv(filename, index_col=None)
        dfs.append(df)
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df['type'] = df_type
    df = df[column_list]
    return df


def extract_momentum(momentum_dict, ticker):
    momentum_dict = ast.literal_eval(momentum_dict)
    if ticker in momentum_dict:
        result = momentum_dict[ticker]
    else:
        result = 1
    return result


def apply_nlp(df):
    df['symbol'] = df['text'].apply(lambda x: process(x, nlp))
    return df


def apply_momentum(df):
    df['momentum'] = df['text'].apply(get_momentum)
    return df


def apply_split(df):
    split_stack = df['symbol'].str.split(',').apply(pd.Series, 1).stack()
    return split_stack


def apply_extract_momentum(df):
    df['momentum'] = df[['momentum', 'symbol']].apply(lambda x: extract_momentum(x.momentum, x.symbol), axis=1)
    return df


def split_symbols_column():
    comments_df = pd.read_csv(os.path.join(PROJECT_PATH, "data/comments_momentum_data/comments_momentum_output_df.csv"), parse_dates=True)
    comment_ticker_split = process_dataframe_in_parallel(comments_df, apply_split)
    comment_ticker_split.index = comment_ticker_split.index.droplevel(-1)
    comment_ticker_split.name = 'symbol'
    del comments_df['symbol']
    comment_ticker_split_df = comments_df.join(comment_ticker_split)
    comments_output_path = os.path.join(PROJECT_PATH, "data/comments_symbol_split_data/comments_symbol_split_output_df.csv")
    comment_ticker_split_df.to_csv(comments_output_path)

    posts_df = pd.read_csv(os.path.join(PROJECT_PATH, "data/posts_momentum_data/posts_momentum_output_df.csv"), parse_dates=True)
    post_ticker_split = posts_df['symbol'].str.split(',').apply(pd.Series, 1).stack()
    post_ticker_split.index = post_ticker_split.index.droplevel(-1)
    post_ticker_split.name = 'symbol'
    del posts_df['symbol']
    post_ticker_split_df = posts_df.join(post_ticker_split)
    posts_output_path = os.path.join(PROJECT_PATH,
                                        "data/posts_symbol_split_data/posts_symbol_split_output_df.csv")
    post_ticker_split_df.to_csv(posts_output_path)


def process_dataframe_in_parallel(df, func):
    df_split = np.array_split(df, cpu_count()-1)
    pool = Pool(cpu_count())
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def save_extracted_momentum():
    comments_split_df = pd.read_csv(
        os.path.join(PROJECT_PATH, "data/comments_symbol_split_data/comments_symbol_split_output_df.csv"),
        parse_dates=True)
    comments_df = process_dataframe_in_parallel(comments_split_df, apply_extract_momentum)
    comments_df = comments_df.drop_duplicates()
    comments_output_path = os.path.join(PROJECT_PATH, "data/comments_extracted_momentum_data/comments_extracted_momentum_output_df.csv")
    comments_df.to_csv(comments_output_path)

    posts_split_df = pd.read_csv(
        os.path.join(PROJECT_PATH, "data/posts_symbol_split_data/posts_symbol_split_output_df.csv"),
        parse_dates=True)
    posts_df = process_dataframe_in_parallel(posts_split_df, apply_extract_momentum)
    posts_df = posts_df.drop_duplicates()
    posts_output_path = os.path.join(PROJECT_PATH,
                                        "data/posts_extracted_momentum_data/posts_extracted_momentum_output_df.csv")
    posts_df.to_csv(posts_output_path)


def save_momentum_df():
    comments_tokenized_df = pd.read_csv(
        os.path.join(PROJECT_PATH, "data/comments_tokenized_data/comments_tokenized_final_output_df.csv"),
        parse_dates=True)
    comments_df = comments_tokenized_df[comments_tokenized_df['symbol'].notnull()]
    comment_momentum_df = process_dataframe_in_parallel(comments_df, apply_momentum)
    comments_momentum_output_path = os.path.join(PROJECT_PATH, "data/comments_momentum_data/comments_momentum_output_df.csv")
    comment_momentum_df.to_csv(comments_momentum_output_path)

    posts_tokenized_df = pd.read_csv(
        os.path.join(PROJECT_PATH, "data/posts_tokenized_data/posts_tokenized_final_output_df.csv"), parse_dates=True)
    posts_df = posts_tokenized_df[posts_tokenized_df['symbol'].notnull()]
    post_momentum_df = process_dataframe_in_parallel(posts_df, apply_momentum)
    posts_momentum_output_path = os.path.join(PROJECT_PATH, "data/posts_momentum_data/posts_momentum_output_df.csv")
    post_momentum_df.to_csv(posts_momentum_output_path)
    

def parse_tokens_from_df(columns, df_type, input_path, output_path):
    df = pd.read_csv(input_path, parse_dates=True)
    df['type'] = df_type
    df = df[columns]
    tokenized_df = process_dataframe_in_parallel(df, apply_nlp)
    tokenized_df.to_csv(output_path)


def save_tokenized_df_approach_2():
    comments_column_list = ['date', 'comment_id', 'text', 'timestamp', 'type']
    comments_input_path = os.path.join(PROJECT_PATH, "data/comments_df.csv")
    comments_output_path = os.path.join(PROJECT_PATH, 'data/comments_tokenized_data/comments_tokenized_final_output_df.csv')
    comments_df_type = "C"
    parse_tokens_from_df(comments_column_list, comments_df_type, comments_input_path, comments_output_path)

    posts_column_list = ['date', 'post_id', 'text', 'timestamp', 'type']
    posts_input_path = os.path.join(PROJECT_PATH, "data/posts_df.csv")
    posts_output_path = os.path.join(PROJECT_PATH, 'data/posts_tokenized_data/posts_tokenized_final_output_df.csv')
    posts_df_type = "P"
    parse_tokens_from_df(posts_column_list, posts_df_type, posts_input_path, posts_output_path)


def ticker_match_df():
    ticker_match_query_string = f"""
     SELECT
     companyid, symbol
     FROM
      "equity-atlas_v3"."shareclassinfo"
    """

    df_ticker_match = pd.DataFrame(dnalab.query(ticker_match_query_string))
    output_path = os.path.join(PROJECT_PATH, "data/ticker_match_db/ticker_match_df.csv")
    df_ticker_match.to_csv(output_path)


def company_match_df():
    company_match_query_string = f"""
     SELECT
     companyid, shortname AS symbol
     FROM
     "equity-atlas_v3"."companyinfo"
    """

    df_company_match = pd.DataFrame(dnalab.query(company_match_query_string))
    output_path = os.path.join(PROJECT_PATH, "data/company_match_db/company_match_df.csv")
    df_company_match.to_csv(output_path)


def save_tokenized_df_approach_1():
    comments_file_chunks = create_smaller_processing_sets(COMMENTS_DATA_PATH)
    posts_file_chunks = create_smaller_processing_sets(POSTS_DATA_PATH)

    comments_column_list = ['comment_id', 'text', 'timestamp', 'type']
    posts_column_list = ['post_id', 'text', 'timestamp', 'type']

    comment_nlp_tokenized_output_path= os.path.join(PROJECT_PATH, 'data/comments_tokenized_data')
    post_nlp_tokenized_output_path = os.path.join(PROJECT_PATH, 'data/posts_tokenized_data')

    comment_nlp_tokenized_processes = []
    post_nlp_tokenized_processes = []

    for comments_files in comments_file_chunks:
        df_chunked = create_df_from_list(comments_files, comments_column_list, "C")
        from_file = '_'.join(comments_files[0].split('/')[-1].split('_')[1:])[:-4]
        to_file = '_'.join(comments_files[-1].split('/')[-1].split('_')[1:])
        output_file_name = f"comments_tokenized_{from_file}_{to_file}"
        output_file_path = os.path.join(comment_nlp_tokenized_output_path, output_file_name)
        p = Process(target=apply_nlp, args=(df_chunked))
        comment_nlp_tokenized_processes.append(p)
        p.start()

    for p in comment_nlp_tokenized_processes:
        p.join()

    for posts_files in posts_file_chunks:
        df_chunked = create_df_from_list(posts_files, posts_column_list, "P")
        from_file = '_'.join(posts_files[0].split('/')[-1].split('_')[1:])[:-4]
        to_file = '_'.join(posts_files[-1].split('/')[-1].split('_')[1:])
        output_file_name = f"posts_tokenized_{from_file}_{to_file}"
        output_file_path = os.path.join(post_nlp_tokenized_output_path, output_file_name)
        p = Process(target=apply_nlp, args=(df_chunked))
        post_nlp_tokenized_processes.append(p)
        p.start()

    for p in post_nlp_tokenized_processes:
        p.join()


if __name__ == '__main__':
    start_time = time.time()

    # These steps should be followed/uncommented in order

    # save_tokenized_df_approach_2()
    # save_momentum_df()
    # split_symbols_column()
    # save_extracted_momentum()

    print('Time taken = {} seconds'.format(time.time() - start_time))
