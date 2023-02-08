from pathlib import Path
import os
from glob import glob
import frontmatter
import logging
logging.basicConfig(encoding='utf-8', level=logging.INFO)

BASE_DIR = "data/tech-blog"
category_dirs = glob(f'{BASE_DIR}/*/', recursive=False)

res = []
required_fields = set(["title", "author", "summary", "date", "tags"])
for d in category_dirs:
    file_list = [i for i in os.listdir(d) if i.endswith('.md')]
    for f in file_list:
        fp = Path(d).joinpath(f)
        fm = frontmatter.load(fp)
        if not fm.metadata or set(fm.metadata.keys()) != required_fields:
            logging.warn(f"{f} doesn't have required fields, skipping...")
            continue
        info = {}
        info.update(fm.metadata)
        info.update(content = fp.read_text())
        res.append(info)

logging.info("Content fetched.")
print(res)

