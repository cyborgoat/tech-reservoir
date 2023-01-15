from pathlib import Path
import os
from glob import glob
import frontmatter
import json

BASE_DIR = "tech-blog"
category_dirs = glob(f'{BASE_DIR}/*/', recursive=False)

res = []
required_fields = set(["title", "author", "summary", "date", "tags"])
for d in category_dirs:
    file_list = [i for i in os.listdir(d) if i.endswith('.md')]
    for f in file_list:
        fm = frontmatter.load(Path(d).joinpath(f))
        if not fm.metadata or set(fm.metadata.keys()) != required_fields:
            continue
        info = {'category': d.split('/')[1]}
        info['slug'] = f.split('.')[0]
        info.update(fm.metadata)
        res.append(info)

with open(Path(BASE_DIR).joinpath("blog_list.json"), 'w', encoding='utf-8') as wf:
    r = json.dumps(res, indent=4, sort_keys=True, default=str)
    wf.write(r)
