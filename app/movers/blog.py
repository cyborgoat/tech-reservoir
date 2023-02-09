#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   blog.py
@Time    :   2023/02/08
@Author  :   Junxiao Guo
@Version :   1.0
@License :   (C)Copyright 2022-2023, Junxiao Guo
@Desc    :   Sync data and send to destination API endpoint
'''
from pathlib import Path
import os
from glob import glob
import frontmatter
import logging
from typing import Dict, List
import requests

logging.basicConfig(encoding='utf-8', level=logging.INFO)

API_HOST = os.environ.get("API_HOST")
API_PORT = os.environ.get("API_PORT")
API_TOKEN = os.environ.get("API_TOKEN")


def blog_list() -> List[Dict]:
    """Fetch blog list"""
    BASE_DIR = "/home/data/tech-blog"
    category_dirs = glob(f'{BASE_DIR}/*/', recursive=False)
    blog_list = []
    required_fields = set(["title", "author", "summary", "date", "tags"])
    for d in category_dirs:
        file_list = [i for i in os.listdir(d) if i.endswith('.md')]
        for f in file_list:
            fp = Path(d).joinpath(f)
            fm = frontmatter.load(fp)

            if not fm.metadata or\
                    set(fm.metadata.keys()) != required_fields:
                continue

            info = {}
            info.update(fm.metadata)
            info.update(content=fp.read_text())
            blog_list.append(info)

    logging.info("Content fetched.")
    return blog_list


def post_blogs():
    """Post blogs to database"""
    url = f"{API_HOST}:{API_PORT}/api/blog/blogs/"
    headers = {"Content-Type": "application/json; charset=utf-8",
               "Authorization": f"Token {API_TOKEN}"}

    data = blog_list()
    for blog in data:
        blog['date'] = blog['date'].strftime('%Y-%m-%d')
        blog['created_on'] = blog['date']
        tags = blog['tags']
        tags_dict = [{"name": tag} for tag in tags]
        blog['tags'] = tags_dict
        response = requests.post(url, headers=headers,
                                 json=blog,
                                 timeout=300)
        print("Status Code", response.status_code)