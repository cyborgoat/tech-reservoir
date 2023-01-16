from pathlib import Path
import pathlib
import os
from glob import glob
import frontmatter
import json
import toml
import logging
logging.basicConfig(encoding='utf-8', level=logging.INFO)

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
        info = {'category': d.split(os.sep)[1]}
        info['slug'] = f.split('.')[0]
        info.update(fm.metadata)
        res.append(info)

# Write blog list
with open(Path(BASE_DIR).joinpath("blog_list.json"), 'w', encoding='utf-8') as wf:
    r = json.dumps(res, indent=4, sort_keys=True, default=str)
    wf.write(r)

# Write config file if not exist
blog_path = Path.cwd().joinpath('tech-blog')
config_path = pathlib.Path.home().joinpath('.tech-reservoir', 'config.toml')
if config_path.exists():
    logging.info("Config file exists, skipping this part")
    logging.info("Content refreshed.")
    exit(0)
config_path.parent.mkdir(parents=True, exist_ok=True)
data = {
    "host": {
        "address": "127.0.0.1",
        "port": 6699,
    },
    "directory": {
        "tech_blog": f"{blog_path}"
    }
}
toml_str = toml.dumps(data)
with open(config_path, 'w', encoding='utf-8') as wf:
    wf.write(toml_str)
logging.info(f"Configuration file has been written to {config_path}")
logging.info("Content refreshed.")
