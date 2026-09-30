"""MkDocs hook: render the code map from docs/codemap.yml and the code's docstrings.

The spec (codemap.yml) gives the story order: stages -> files -> functions and the
arrows between them. Every description shown on the site is the first line of the
function's docstring, read from the source at build time, so the website can never
drift from the comments in the code.

In any page, a line `{{ codemap }}` expands to the full map and
`{{ codemap:<stage-id> }}` to a single stage, `{{ codemap:overview }}` to the
file-level diagram only. `{{ hardcoded }}` lists every `# HARDCODED:` comment.
"""

import ast
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "docs" / "codemap.yml"
MARKER = re.compile(r"^\{\{\s*codemap(?::([\w-]+))?\s*\}\}\s*$", re.M)
HARDCODED = re.compile(r"^\{\{\s*hardcoded\s*\}\}\s*$", re.M)
SCAN = ["kirad", "scripts", "research"]


def _index(path: str) -> dict:
    """Map 'func' and 'Class.method' to (lineno, first docstring line) for one file."""
    tree = ast.parse((ROOT / path).read_text())
    out = {}

    def visit(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                doc = (ast.get_docstring(child) or "").strip().splitlines()
                out[name] = (child.lineno, doc[0].strip() if doc else "")
                if isinstance(child, ast.ClassDef):
                    visit(child, f"{name}.")

    visit(tree)
    doc = (ast.get_docstring(tree) or "").strip().splitlines()
    out["__module__"] = (1, doc[0].strip() if doc else "")
    return out


def _nid(path: str, func: str = "") -> str:
    return re.sub(r"\W", "_", f"{path}__{func}")


def _esc(text: str) -> str:
    return text.replace('"', "'").replace("<", "‹").replace(">", "›")


def _label(path: str) -> str:
    """Short box title: file name, or 'folder/model.py' for experiment files."""
    p = Path(path)
    if p.parts[0] == "research":
        return f"{p.parent.name.replace('ordinal_regression_', '')}/{p.name}"
    return p.name


def _short(text: str, n: int = 70) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


class CodeMap:
    def __init__(self):
        self.spec = yaml.safe_load(SPEC.read_text())
        self.url = self.spec["repo_url"].rstrip("/")
        self.cache = {}

    def info(self, path, func):
        if path not in self.cache:
            self.cache[path] = _index(path)
        if func not in self.cache[path]:
            raise KeyError(f"codemap.yml: {func!r} not found in {path}")
        return self.cache[path][func]

    def link(self, path, func="__module__"):
        line, _ = self.info(path, func)
        return f"{self.url}/{path}#L{line}"

    def overview(self) -> str:
        """File-level diagram: one box per file per stage, story arrows between them."""
        lines = ["```mermaid", "flowchart TB"]
        for stage in self.spec["stages"]:
            lines.append(f'  subgraph {stage["id"]}["{_esc(stage["title"])}"]')
            lines.append("    direction LR")
            for f in stage["files"]:
                _, mod = self.info(f["path"], "__module__")
                label = f"<b>{_label(f['path'])}</b><br/>{_esc(_short(mod, 48))}"
                lines.append(f'    {_nid(stage["id"] + f["path"])}["{label}"]')
            lines.append("  end")
        for e in self.spec.get("file_edges", []):
            (sa, pa), (sb, pb) = (x.split(":", 1) for x in (e["from"], e["to"]))
            lab = f'|"{_esc(e["label"])}"|' if e.get("label") else ""
            lines.append(f"  {_nid(sa + pa)} -->{lab} {_nid(sb + pb)}")
        for stage in self.spec["stages"]:
            for f in stage["files"]:
                nid = _nid(stage["id"] + f["path"])
                lines.append(f'  click {nid} "{self.link(f["path"])}" _blank')
        lines.append("```")
        return "\n".join(lines)

    def stage(self, sid: str) -> str:
        """Function-level diagram for one stage, plus a table with links."""
        stage = next(s for s in self.spec["stages"] if s["id"] == sid)
        out = [f"### {stage['title']}", "", stage.get("story", "").strip(), ""]
        out += ["```mermaid", "flowchart TB"]
        for f in stage["files"]:
            out.append(f'  subgraph {_nid(f["path"])}_box["{_label(f["path"])}"]')
            for fn in f.get("functions", []):
                _, doc = self.info(f["path"], fn)
                label = f"<b>{fn}()</b><br/>{_esc(_short(doc))}"
                out.append(f'    {_nid(f["path"], fn)}["{label}"]')
            out.append("  end")
        for a, b, *lab in stage.get("flow", []):
            pa, fa = a.split("::")
            pb, fb = b.split("::")
            label = f'|"{_esc(lab[0])}"|' if lab else ""
            out.append(f"  {_nid(pa, fa)} -->{label} {_nid(pb, fb)}")
        for f in stage["files"]:
            for fn in f.get("functions", []):
                out.append(f'  click {_nid(f["path"], fn)} "{self.link(f["path"], fn)}" _blank')
        out += ["```", "", "| Function | File | What it does |", "|---|---|---|"]
        for f in stage["files"]:
            for fn in f.get("functions", []):
                line, doc = self.info(f["path"], fn)
                out.append(
                    f"| [`{fn}`]({self.link(f['path'], fn)}) | `{f['path']}:{line}` | {doc or '—'} |"
                )
        return "\n".join(out) + "\n"

    def render(self, sid):
        if sid == "overview":
            return self.overview() + "\n"
        if sid:
            return self.stage(sid)
        parts = ["## The whole story, file by file", "", self.overview(), ""]
        parts += [self.stage(s["id"]) for s in self.spec["stages"]]
        return "\n".join(parts)


def hardcoded_table(url: str) -> str:
    """Every `# HARDCODED:` comment in the code, as a table with links."""
    rows = ["| Where | What |", "|---|---|"]
    for folder in SCAN:
        for path in sorted((ROOT / folder).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if "# HARDCODED:" in line:
                    what = line.split("# HARDCODED:", 1)[1].strip().replace("|", "\\|")
                    rows.append(f"| [`{rel}:{i}`]({url}/{rel}#L{i}) | {what} |")
    return "\n".join(rows) + "\n"


def on_page_markdown(markdown, page, config, files):
    if "{{" not in markdown:
        return markdown
    if HARDCODED.search(markdown):
        url = yaml.safe_load(SPEC.read_text())["repo_url"].rstrip("/")
        markdown = HARDCODED.sub(lambda m: hardcoded_table(url), markdown)
    if "codemap" in markdown:
        cm = CodeMap()
        markdown = MARKER.sub(lambda m: cm.render(m.group(1)), markdown)
    return markdown
