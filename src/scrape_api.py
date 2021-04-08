import requests as re
import json
import time
import pandas as pd
import glob
import os
from datetime import datetime

PROJECT_PATH = os.getcwd()
POSTS_PATH = 'data/reddit_posts'
COMMENTS_PATH = 'data/reddit_comments'
POSTS_DATA_PATH = os.path.join(PROJECT_PATH, POSTS_PATH)
COMMENTS_DATA_PATH = os.path.join(PROJECT_PATH, COMMENTS_PATH)


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
                post_id = post["id"] if "id" in post.keys() else None
                text = post["selftext"] if "selftext" in post.keys() else None
                title = post["title"] if "title" in post.keys() else None
                ratio = post["upvote_ratio"] if "upvote_ratio" in post.keys(
                ) else None
                date = post["created_utc"] if "created_utc" in post.keys(
                ) else None
                author = post["author"] if "author" in post.keys() else None
                post_item = {"post_id": post_id, "text": text, "title": title,
                             "upvote_ratio": ratio, "timestamp": date, "author": author}
                posts_list.append(post_item)
            post_df = pd.DataFrame(posts_list)
            filename = f"df_{str(start)}.csv"
            filepath = os.path.join(POSTS_DATA_PATH, filename)
            post_df.to_csv(filepath)
            last_post = posts[-1]
            start = last_post['created_utc']
            time.sleep(1)
        except Exception as e:
            print(res, start, e, len(posts))


def get_posts_df():
    pd.set_option('display.max_columns', None)
    all_files = glob.glob(POSTS_DATA_PATH + "/*.csv")

    dfs = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None)
        dfs.append(df)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    non_nan_df = df[df['text'].notna()]
    filtered_df = non_nan_df[(non_nan_df['text'] != "[removed]") & (
        non_nan_df['text'] != '[deleted]')]
    filtered_df['date'] = filtered_df['timestamp'].apply(
        lambda x: datetime.fromtimestamp(x)).tolist()
    filtered_df.set_index('date', inplace=True)
    final_df = filtered_df.drop(filtered_df.columns[0], axis=1)
    final_df = final_df.sort_index()
    path = os.path.join(os.path.join(PROJECT_PATH, 'data'), 'posts_df.csv')
    final_df.to_csv(path)


def get_posts_list():
    path = os.path.join(os.path.join(PROJECT_PATH, 'data'), 'posts_df.csv')
    df = pd.read_csv(path, index_col='date', parse_dates=True)
    posts = df['post_id'].tolist()
    return posts


def get_comments(post_ids):
    for post_id in post_ids:
        start = 1614556800
        end = 1617235199
        res = None
        comments = None
        while start < end:
            url = F"https://api.pushshift.io/reddit/search/comment/?link_id={post_id}&sort=asc&sort_type=created_utc&user_removed=False&mod_removed=False&after={start}&is_video=False&size=100"
            try:
                res = re.get(url)
                if res.status_code != 200 or res == None:
                    break
                comments = json.loads(res.text)['data']
                if len(comments) == 0:
                    break
                comment_list = []
                for comment in comments:
                    comment_id = comment["id"] if "id" in comment.keys(
                    ) else None
                    text = comment["body"] if "body" in comment.keys(
                    ) else None
                    date = comment["created_utc"] if "created_utc" in comment.keys(
                    ) else None
                    author = comment["author"] if "author" in comment.keys(
                    ) else None
                    score = comment["score"] if "score" in comment.keys(
                    ) else None
                    awards = comment["total_awards_received"] if "total_awards_received" in comment.keys(
                    ) else None
                    comment_item = {"comment_id": comment_id, "text": text, "score": score,
                                    "awards": awards, "timestamp": date, "author": author}
                    comment_list.append(comment_item)
                comment_df = pd.DataFrame(comment_list)
                filename = f"df_{str(post_id)}_{str(start)}.csv"
                filepath = os.path.join(COMMENTS_DATA_PATH, filename)
                comment_df.to_csv(filepath)
                last_comment = comments[-1]
                start = last_comment['created_utc']
                time.sleep(1)
            except Exception as e:
                print(res, start, e, post_id, len(comments))


if __name__ == '__main__':
    posts = get_posts_list()
    posts.sort()
    post_id = 'lwgdok'
    idx = posts.index(post_id)
    print(idx)
    #get_comments(posts[1235:])
