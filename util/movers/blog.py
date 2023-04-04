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
from glob import glob
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import os
import logging
import requests
import frontmatter
import datetime

logging.basicConfig(encoding='utf-8', level=logging.INFO)


def blog_list(config: Dict) -> List[Dict]:
    """Fetch blog list"""
    data = []

    category_dirs = glob(f'{config["BLOG_DIR"]}/*/', recursive=False)
    required_fields = set(["title", "author", "summary", "date", "tags"])

    for cat_dir in category_dirs:
        file_list = [i for i in os.listdir(cat_dir) if i.endswith('.md')]
        for file in file_list:
            filepath = Path(cat_dir).joinpath(file)
            front_matter = frontmatter.load(filepath)

            if not front_matter.metadata or\
                    set(front_matter.metadata.keys()) != required_fields:
                continue

            info = {}
            info.update(front_matter.metadata)
            info.update(content=filepath.read_text(encoding='utf-8').replace(
                "../../assets", "https://github.com/cyborgoat/tech-reservoir/blob/main/assets").replace(".png)", ".png?raw=true)"))
            data.append(info)
    return data


def post_blogs(config: Dict):
    """Post blogs to database"""
    url = f"{config['API_HOST']}/api/blog/blogs/"
    headers = {"Content-Type": "application/json; charset=utf-8",
               "Authorization": f"Token {config['API_TOKEN']}"}
    data = blog_list(config)
    pbar = tqdm(data)
    for blog in pbar:
        pbar.set_description("Processing %s" % blog.get('title'))
        blog['date'] = blog.pop(
            'date', datetime.date.today()).strftime('%Y-%m-%d')
        blog['tags'] = [{"name": tag} for tag in blog.get('tags', [])]
        response = requests.post(url, headers=headers,
                                 json=blog,
                                 timeout=300)
        if response.status_code != 201:
            logging.error("%s failed to post",
                          blog.get("title", "Unknown Blog"))
