from __future__ import annotations
def save_list(lines, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for x in lines:
            f.write(str(x).rstrip() + "\n")
