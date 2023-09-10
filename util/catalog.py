import json
import os
import pathlib
from glob import glob
from typing import List, Dict
from pathlib import Path
import frontmatter

BLOG_DIR = pathlib.Path(__file__).parents[1].joinpath("tech-blog")
ASSETS_DIR = pathlib.Path(__file__).parents[1].joinpath("assets")


def write_to_json():
    """Fetch blog list"""
    catalog = []

    category_dirs = glob(str(BLOG_DIR) + '/*/', recursive=False)
    required_fields = {"title", "author", "summary", "date", "tags"}

    for cat_dir in category_dirs:
        file_list = [i for i in os.listdir(cat_dir) if i.endswith('.md')]
        for raw_fp in file_list:
            filepath = Path(cat_dir).joinpath(raw_fp)
            front_matter = frontmatter.load(filepath)

            if not front_matter.metadata or \
                    set(front_matter.metadata.keys()) != required_fields:
                continue

            info = {}
            info.update(front_matter.metadata)
            category = cat_dir.split('/')[-2]
            file_name = raw_fp.replace('.md', '.json')
            ASSETS_DIR.joinpath(category).mkdir(parents=True, exist_ok=True)

            catalog_item = {'category': category, 'fname': file_name}
            catalog_item.update(front_matter.metadata)
            catalog.append(catalog_item)

            # Update content
            info.update(content=filepath.read_text(encoding='utf-8').replace("../../assets",
                                                                             "https://github.com/cyborgoat/tech-reservoir/blob/main/assets").replace(
                ".png)",
                ".png?raw=true)"))

            with open(ASSETS_DIR.joinpath(category, file_name), 'w', encoding='utf-8') as f:
                json.dump(info, f, default=str)

        with open(ASSETS_DIR.joinpath("catalog.json"), 'w', encoding='utf-8') as f:
            json.dump(catalog, f, default=str)


if __name__ == '__main__':
    write_to_json()
