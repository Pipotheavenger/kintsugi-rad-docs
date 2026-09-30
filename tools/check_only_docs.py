"""Fail if any Python file changed beyond docstrings and comments.

Compares each tracked .py file against a git ref (default: the upstream-base tag)
after stripping docstrings. Comments never reach the AST, so any remaining
difference is a code change.

Usage: python tools/check_only_docs.py [git-ref] [file ...]
"""

import ast
import subprocess
import sys

DOC_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def strip_docstrings(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, DOC_OWNERS) and node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


def main() -> int:
    ref = sys.argv[1] if len(sys.argv) > 1 else "upstream-base"
    files = sys.argv[2:] or subprocess.run(
        ["git", "ls-files", "*.py"], capture_output=True, text=True, check=True
    ).stdout.split()
    failed = []
    for path in files:
        if path.startswith(("tools/", "docs/")):
            continue  # documentation tooling, not upstream code
        old = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True)
        if old.returncode != 0:
            continue  # file is new relative to ref
        with open(path) as f:
            new = f.read()
        try:
            if strip_docstrings(old.stdout) != strip_docstrings(new):
                failed.append(path)
        except SyntaxError as e:
            failed.append(f"{path} (syntax error: {e})")
    for path in failed:
        print(f"CODE CHANGED: {path}")
    if not failed:
        print(f"OK: {len(files)} files, only docstrings/comments differ from {ref}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
