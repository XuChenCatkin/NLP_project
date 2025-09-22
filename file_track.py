
import ast
import json
import os
from collections import defaultdict
from typing import Dict, List

ROOT = os.path.abspath(os.getcwd())

FILE_EXTS = {".py"}

IO_FUNCS = {
    "open",
    "pandas.read_csv", "pandas.read_json", "pandas.read_parquet",
    "pd.read_csv", "pd.read_json", "pd.read_parquet",
    "json.load", "json.dump",
}

KEY_IMPORTS = [
    "torch", "transformers", "faiss", "faiss_cpu", "sklearn",
    "spacy", "typer", "argparse", "pandas", "numpy", "networkx",
]

def analyze_py(path: str) -> Dict:
    info = {
        "path": path,
        "lines": 0,
        "has_main": False,
        "functions": [],
        "classes": [],
        "imports": [],
        "io_calls": [],
        "argparse": False,
        "typer": False,
        "uses_cuda": False,
    }
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        info["lines"] = src.count("\\n") + 1
        tree = ast.parse(src)
    except Exception as e:
        info["error"] = str(e)
        return info

    # scan AST
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            info["functions"].append(node.name)
        elif isinstance(node, ast.ClassDef):
            info["classes"].append(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.split(".")[0]
                info["imports"].append(mod)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                info["imports"].append(node.module.split(".")[0])

    # heuristics
    src_lower = src.lower()
    info["has_main"] = "__name__ == '__main__'" in src or '__name__ == "__main__"' in src
    info["argparse"] = "import argparse" in src or "argparse." in src
    info["typer"] = "import typer" in src or "typer." in src
    info["uses_cuda"] = "cuda" in src_lower or "torch.cuda" in src_lower

    # crude IO detection
    for key in IO_FUNCS:
        if key in src:
            info["io_calls"].append(key)

    # normalize imports (unique, filtered)
    info["imports"] = sorted(set([m for m in info["imports"] if m]))
    return info

def walk_repo(root: str) -> List[Dict]:
    results = []
    for dirpath, _, filenames in os.walk(root):
        # skip common virtualenv/hidden
        if any(skip in dirpath for skip in (".git", ".venv", "__pycache__", "build", "dist")):
            continue
        for name in filenames:
            ext = os.path.splitext(name)[1]
            if ext in FILE_EXTS:
                results.append(analyze_py(os.path.join(dirpath, name)))
    return results

def main():
    out = walk_repo(ROOT)
    # save JSON and a compact Markdown table
    with open("audit_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    headers = ["path","lines","has_main","argparse","typer","uses_cuda","imports","functions","classes","io_calls"]
    def row(item):
        return "|" + "|".join([
            item.get("path",""),
            str(item.get("lines",0)),
            str(item.get("has_main",False)),
            str(item.get("argparse",False)),
            str(item.get("typer",False)),
            str(item.get("uses_cuda",False)),
            ",".join(item.get("imports",[]))[:80],
            ",".join(item.get("functions",[]))[:80],
            ",".join(item.get("classes",[]))[:80],
            ",".join(item.get("io_calls",[]))[:80],
        ]) + "|"

    md = ["# Repo Audit Report", "", "|" + "|".join(headers) + "|", "|" + "|".join(["---"]*len(headers)) + "|"]
    for item in out:
        md.append(row(item))

    with open("audit_report.md", "w", encoding="utf-8") as f:
        f.write("\\n".join(md))

    print("Wrote audit_report.json and audit_report.md")

if __name__ == "__main__":
    main()

