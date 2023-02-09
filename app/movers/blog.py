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


logging.basicConfig(encoding='utf-8', level=logging.INFO)

host = os.environ.get("API_HOST")
port = os.environ.get("API_PORT")
token = os.environ.get("API_TOKEN")


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
