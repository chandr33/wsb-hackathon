import requests as re
import json
import time
import pandas as pd
import glob
import os
from datetime import datetime

PROJECT_PATH= os.path.dirname(os.getcwd())
POSTS_PATH = 'data/reddit_posts'
DATA_PATH = os.path.join(PROJECT_PATH, POSTS_PATH)


def get_posts():
    start = 1614556800
    end = 1617235199
    res = None
    posts = None
    while start < end:
        url = F"https://api.pushshift.io/reddit/search/submission/?subreddit=wallstreetbets&sort=asc&sort_type=created_utc&after={start}&is_video=False&size=100"
        try:
            res = re.get(url)
            posts = json.loads(res.text)['data']
            posts_list = []
            for post in posts:
                post_id  = post["id"] if "id" in post.keys() else None
                text = post["selftext"] if "selftext" in post.keys() else None
                title = post["title"] if "title" in post.keys() else None
                ratio = post["upvote_ratio"] if "upvote_ratio" in post.keys() else None
                date = post["created_utc"] if "created_utc" in post.keys() else None
                author = post["author"] if "author" in post.keys() else None
                post_item = {"post_id": post_id, "text": text, "title": title, "upvote_ratio": ratio, "timestamp": date, "author": author}
                posts_list.append(post_item)
            post_df = pd.DataFrame(posts_list)
            filename = f"df_{str(start)}.csv"
            filepath = os.path.join(DATA_PATH, filename)
            post_df.to_csv(filepath)
            last_post = posts[-1]
            start = last_post['created_utc']
            time.sleep(1)
        except Exception as e:
            print(res, start, e, len(posts))


def get_df():
    pd.set_option('display.max_columns', None)
    all_files = glob.glob(DATA_PATH + "/*.csv")

    dfs = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None)
        dfs.append(df)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    non_nan_df = df[df['text'].notna()]
    filtered_df = non_nan_df[(non_nan_df['text'] != "[removed]") & (non_nan_df['text'] != '[deleted]')]
    filtered_df['date'] = filtered_df['timestamp'].apply(lambda x: datetime.fromtimestamp(x)).tolist()
    filtered_df.set_index('date', inplace=True)
    final_df = filtered_df.drop(filtered_df.columns[0], axis=1)
    final_df = final_df.sort_index()
    final_df.to_csv('posts_df.csv')


def get_comments():
    # TODO - Get comments
    url = F"https://api.pushshift.io/reddit/search/comment/?subreddit=wallstreetbets&id=lw72rs&sort=asc&sort_type=created_utc&size=100"
    res = re.get(url)
    comments = json.loads(res.text)['data']
    print(len(comments))
    for k, v in comments[-1].items():
        print(f"{k} -> {v}")
    pass


if __name__ == '__main__':
    get_df()
