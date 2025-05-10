#!/usr/bin/env python3
"""
Batch-refactor front matter of all markdown posts under `posts/` to match the style of `standard_post.md`.
Renames `summary` to `excerpt`, ensures quoted values, inline `tags` array, and consistent key order.
"""
import os

BASE_DIR = os.path.join(os.path.dirname(__file__), 'posts')
KEY_ORDER = ['title', 'date', 'author', 'authorImage', 'tags', 'excerpt', 'video']

def parse_front_matter(lines):
    fm = {}
    i = 1
    while i < len(lines):
        line = lines[i].rstrip('\n')
        if line.strip() == '---':
            break
        if ':' in line:
            key, val = line.split(':', 1)
            key = key.strip()
            val = val.strip().strip('"')
            if key == 'tags' and not val:
                tags = []
                i += 1
                while i < len(lines) and lines[i].startswith('  -'):
                    tags.append(lines[i].split('-',1)[1].strip())
                    i += 1
                fm['tags'] = ','.join(tags)
                continue
            fm[key] = val
        i += 1
    return fm, i + 1


def format_tags(raw):
    tags = [t.strip() for t in raw.split(',') if t.strip()]
    return '[' + ', '.join(f'"{t}"' for t in tags) + ']'


def build_front_matter(meta):
    lines = ['---']
    for key in KEY_ORDER:
        if key not in meta or not meta[key]:
            continue
        val = meta[key]
        if key == 'tags':
            line = f"tags: {format_tags(val)}"
        else:
            line = f'{key}: "{val}"'
        lines.append(line)
    lines.append('---\n')
    return [l + '\n' for l in lines]


def process_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if not lines or not lines[0].strip() == '---':
        return
    fm, content_start = parse_front_matter(lines)
    if 'summary' in fm:
        fm['excerpt'] = fm.pop('summary')
    new_fm = build_front_matter(fm)
    content = lines[content_start:]
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(new_fm + content)
    print(f"Refactored: {path}")


def main():
    for root, _, files in os.walk(os.path.abspath(BASE_DIR)):
        for fn in files:
            if fn.endswith('.md'):
                process_file(os.path.join(root, fn))

if __name__ == '__main__':
    main()
