"""MkDocs hook: render the code map from docs/codemap.yml and the code's docstrings.

The spec (codemap.yml) gives the story order: stages -> files -> functions and the
arrows between them. Every description shown on the site is the first line of the
function's docstring, read from the source at build time, so the website can never
drift from the comments in the code.

Markers (each alone on a line):
  {{ codemap }}            every stage, as numbered paths
  {{ codemap:<stage-id> }} one stage
  {{ codemap:overview }}   one row of clickable stages
  {{ tour }}               the step-by-step walkthrough from docs/tour.yml
  {{ files }}              one card per file: functions, calls, called by
  {{ hardcoded }}          every `# HARDCODED:` comment
"""

import ast
import html
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "docs" / "codemap.yml"
TOUR = ROOT / "docs" / "tour.yml"
TOUR_RE = re.compile(r"^\{\{\s*tour\s*\}\}\s*$", re.M)
FILES_RE = re.compile(r"^\{\{\s*files\s*\}\}\s*$", re.M)
WALK = "../walkthrough/"  # relative link from code-map/ and files/ pages
CLASSDEFS = [
    "  classDef main fill:#FFF4DC,stroke:#A67C1B,stroke-width:2px,color:#1A1A1A",
    "  classDef opt fill:#F3F4F6,stroke:#B8BEC7,color:#555,stroke-dasharray:4 3",
    "  classDef edge fill:#FFFFFF,stroke:#1F7A5C,color:#1F7A5C,stroke-width:1.5px",
]
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


def _cell(text: str) -> str:
    """Markdown table cell: escape <placeholders> and pipes so they survive rendering."""
    parts = re.split(r"(`[^`]*`)", text)  # leave code spans as-is
    out = "".join(
        x if x.startswith("`") else html.escape(x, quote=False).replace("[", "\\[").replace("]", "\\]")
        for x in parts
    )  # escaped brackets: shapes like [ΣW, 80] must not become Markdown links
    return out.replace("|", "\\|")


def _wrap(text: str, width: int = 34) -> str:
    """Break a description into short lines so diagram boxes stay narrow."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "<br/>".join(lines)


def _kmap(lines) -> str:
    """Raw HTML block rendered by docs/assets/codemap.js at natural size, clickable."""
    return f'<div class="kmap">\n{html.escape(chr(10).join(lines))}\n</div>'


def _label(path: str) -> str:
    """Short box title: file name, or 'folder/model.py' for experiment files."""
    p = Path(path)
    if p.parts[0] == "research":
        return f"{p.parent.name.replace('ordinal_regression_', '')}/{p.name}"
    return p.name


class CodeMap:
    def __init__(self):
        self.spec = yaml.safe_load(SPEC.read_text())
        self.url = self.spec["repo_url"].rstrip("/")
        self.cache = {}
        self.tour = yaml.safe_load(TOUR.read_text()) if TOUR.exists() else {"steps": []}
        self.steps = self.tour["steps"]
        for i, st in enumerate(self.steps, 1):
            st["n"] = i
            self.info(st["file"], st["function"])  # fail the build on a wrong name
        self.step_of = {(st["file"], st["function"]): st["n"] for st in self.steps}
        self.titles = {s["id"]: s["title"] for s in self.spec["stages"]}

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
        """One row of stages, left to right; each box links to its stage."""
        lines = ["flowchart LR"]
        prev = None
        for stage in self.spec["stages"]:
            files = " · ".join(_label(f["path"]) for f in stage["files"])
            nums = [st["n"] for st in self.steps if st["stage"] == stage["id"]]
            steps = f"<br/><i>steps {nums[0]}–{nums[-1]}</i>" if nums else "<br/><i>optional</i>"
            label = f"<b>{_esc(stage['title'])}</b><br/>{_wrap(_esc(files), 26)}{steps}"
            sid = f"st_{stage['id']}"
            lines.append(f'  {sid}["{label}"]')
            lines.append(f"  class {sid} {'main' if nums else 'opt'}")
            if prev and nums:
                lines.append(f"  {prev} ==> {sid}")
            elif prev:
                lines.append(f"  {prev} -.-> {sid}")
            if nums:
                prev = sid
            lines.append(f'  click {sid} "code-map/#stage-{stage["id"]}"')
        return _kmap(lines + CLASSDEFS)

    def stage(self, sid: str) -> str:
        """Stage diagram: numbered main path (gold, thick) + optional functions (grey)."""
        stage = next(s for s in self.spec["stages"] if s["id"] == sid)
        main = [st for st in self.steps if st["stage"] == sid]
        out = [f"### {stage['title']} {{#stage-{sid}}}", "", stage.get("story", "").strip(), ""]
        if main:
            out.append(
                f"*Main path: steps {main[0]['n']}–{main[-1]['n']} of a `train` run "
                f"([walkthrough]({WALK}#step-{main[0]['n']})). Gold boxes open the step; "
                "grey boxes are side paths and open the code.*"
            )
            out.append("")
        # every function of the stage, grouped by file, in codemap order + tour extras
        groups = {}
        for f in stage["files"]:
            groups.setdefault(f["path"], []).extend(f.get("functions", []))
        for st in main:
            if st["function"] not in groups.setdefault(st["file"], []):
                groups[st["file"]].append(st["function"])
        diagram = ["flowchart LR"]
        for path, fns in groups.items():
            if not fns:
                continue
            diagram.append(f'  subgraph {_nid(path)}_box["{_label(path)}"]')
            diagram.append("    direction TB")
            for fn in fns:
                _, doc = self.info(path, fn)
                n = self.step_of.get((path, fn)) if (path, fn) in {(m["file"], m["function"]) for m in main} else None
                head = f"{n} · {fn}()" if n else f"{fn}()"
                nid = _nid(path, fn)
                diagram.append(f'    {nid}["<b>{_esc(head)}</b><br/>{_wrap(_esc(doc))}"]')
                diagram.append(f"    class {nid} {'main' if n else 'opt'}")
                target = f"{WALK}#step-{n}" if n else self.link(path, fn)
                diagram.append(f'    click {nid} "{target}"' + ("" if n else " _blank"))
            diagram.append("  end")
        # main path arrows, plus where it comes from and where it goes next
        if main:
            first, last = main[0], main[-1]
            if first["n"] > 1:
                p = self.steps[first["n"] - 2]
                diagram.append(f'  from_prev(["◀ from step {p["n"]}: {_esc(p["function"])}()"])')
                diagram.append("  class from_prev edge")
                diagram.append(f"  from_prev ==> {_nid(first['file'], first['function'])}")
                diagram.append(f'  click from_prev "#stage-{p["stage"]}"')
            for a, b in zip(main, main[1:]):
                diagram.append(f"  {_nid(a['file'], a['function'])} ==> {_nid(b['file'], b['function'])}")
            if last["n"] < len(self.steps):
                nx = self.steps[last["n"]]
                diagram.append(f'  to_next(["▶ next, step {nx["n"]}: {_esc(nx["function"])}()"])')
                diagram.append("  class to_next edge")
                diagram.append(f"  {_nid(last['file'], last['function'])} ==> to_next")
                diagram.append(f'  click to_next "#stage-{nx["stage"]}"')
        seq = {(a["file"], a["function"], b["file"], b["function"]) for a, b in zip(main, main[1:])}
        for a, b, *lab in stage.get("flow", []):
            pa, fa = a.split("::")
            pb, fb = b.split("::")
            if (pa, fa, pb, fb) in seq or (pb, fb, pa, fa) in seq:
                continue
            label = f'|"{_esc(lab[0])}"|' if lab else ""
            diagram.append(f"  {_nid(pa, fa)} -.->{label} {_nid(pb, fb)}")
        out += [_kmap(diagram + CLASSDEFS), "", "| Step | Function | File | What it does |", "|---|---|---|---|"]
        for path, fns in groups.items():
            for fn in fns:
                line, doc = self.info(path, fn)
                n = self.step_of.get((path, fn))
                step = f"[{n}]({WALK}#step-{n})" if n and any(m["n"] == n for m in main) else "—"
                out.append(
                    f"| {step} | [`{fn}`]({self.link(path, fn)}) | `{path}:{line}` | {_cell(doc) or '—'} |"
                )
        return "\n".join(out) + "\n"

    def tour_page(self) -> str:
        """Numbered path of the whole run, then one section per step with prev/next."""
        lines = ["flowchart TB"]
        order = []
        for stage in self.spec["stages"]:
            steps = [st for st in self.steps if st["stage"] == stage["id"]]
            if not steps:
                continue
            lines.append(f'  subgraph t_{stage["id"]}["{_esc(stage["title"])}"]')
            lines.append("    direction LR")
            for st in steps:
                lines.append(f'    s{st["n"]}["<b>{st["n"]}</b> · {_esc(st["function"])}()"]')
                lines.append(f"    class s{st['n']} main")
                lines.append(f'    click s{st["n"]} "#step-{st["n"]}"')
                order.append(st["n"])
            lines.append("  end")
        for a, b in zip(order, order[1:]):
            lines.append(f"  s{a} ==> s{b}")
        out = [self.tour.get("intro", "").strip(), "", _kmap(lines + CLASSDEFS), ""]
        n_all = len(self.steps)
        for st in self.steps:
            n, path, fn = st["n"], st["file"], st["function"]
            line, doc = self.info(path, fn)
            out += [
                f"### Step {n} · `{fn}` {{#step-{n}}}",
                "",
                f"`{path}:{line}` · [open code ↗]({self.link(path, fn)}) · "
                f"stage [{_esc(self.titles[st['stage']])}](../code-map/#stage-{st['stage']})",
                "",
                f"**Does:** {_cell(doc) or '—'}",
                "",
                "| In | Out |",
                "|---|---|",
                f"| {_cell(st.get('in', '—'))} | {_cell(st.get('out', '—'))} |",
                "",
            ]
            if st.get("note"):
                out += [f"!!! note \"Watch out\"", f"    {st['note'].strip()}", ""]
            nav = []
            if n > 1:
                p = self.steps[n - 2]
                nav.append(f"[← Step {n - 1} · `{p['function']}`](#step-{n - 1})")
            if n < n_all:
                q = self.steps[n]
                nav.append(f"[Step {n + 1} · `{q['function']}` →](#step-{n + 1})")
            out += [" &nbsp;&nbsp;|&nbsp;&nbsp; ".join(nav), "", "---", ""]
        return "\n".join(out)

    def render(self, sid):
        if sid == "overview":
            return self.overview() + "\n"
        if sid:
            return self.stage(sid)
        parts = ["## Stages", "", self.overview().replace('"code-map/#', '"#'), ""]
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
                    what = line.split("# HARDCODED:", 1)[1].strip()
                    rows.append(f"| [`{rel}:{i}`]({url}/{rel}#L{i}) | {_cell(what)} |")
    return "\n".join(rows) + "\n"


GENERIC = {
    "forward", "step", "update", "compute", "build", "eval", "run", "get", "log", "numpy",
    "reset", "item", "load_state_dict", "state_dict", "cuda", "train", "to", "mean", "sum",
    "__init__", "__call__", "__getitem__", "__len__", "__iter__", "info", "split", "cost",
}


def _defs():
    """All top-level functions/classes and methods in the code: qualname -> (path, node)."""
    out = {}
    for folder in SCAN:
        for path in sorted((ROOT / folder).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    out[(rel, node.name)] = node
                    if isinstance(node, ast.ClassDef):
                        for m in node.body:
                            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                out[(rel, f"{node.name}.{m.name}")] = m
    return out


def call_graph():
    """Approximate static call graph: (path, qualname) -> set of (path, qualname) it calls."""
    defs = _defs()
    by_name = {}
    for (path, q) in defs:
        by_name.setdefault(q.split(".")[-1], []).append((path, q))
    graph = {k: set() for k in defs}
    for (path, q), node in defs.items():
        cls = q.split(".")[0] if "." in q else None
        body = node.body if not isinstance(node, ast.ClassDef) else []
        for sub in (n for b in body for n in ast.walk(b)):
            if not isinstance(sub, ast.Call):
                continue
            f = sub.func
            target = None
            if isinstance(f, ast.Name):
                cands = [c for c in by_name.get(f.id, []) if "." not in c[1]]
                same = [c for c in cands if c[0] == path]
                target = (same or cands)[0] if (same or len(cands) == 1) else None
            elif isinstance(f, ast.Attribute):
                if isinstance(f.value, ast.Name) and f.value.id == "self" and cls:
                    key = (path, f"{cls}.{f.attr}")
                    target = key if key in defs else None
                if target is None and f.attr not in GENERIC:
                    cands = by_name.get(f.attr, [])
                    target = cands[0] if len(cands) == 1 else None
            if target and target != (path, q):
                graph[(path, q)].add(target)
    return defs, graph


def files_page(cm) -> str:
    """One card per file (story order): functions with step, calls, called by."""
    defs, graph = call_graph()
    callers = {}
    for src, dsts in graph.items():
        for d in dsts:
            callers.setdefault(d, set()).add(src)
    order = []
    for stage in cm.spec["stages"]:
        for f in stage["files"]:
            if f["path"] not in order:
                order.append(f["path"])
    for folder in SCAN:
        for path in sorted((ROOT / folder).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            if rel not in order and any(k[0] == rel for k in defs):
                order.append(rel)

    def ref(k):
        return f"[`{k[1]}`]({cm.url}/{k[0]}#L{defs[k].lineno})" + ("" if k[0] == cur else f" <small>{_label(k[0])}</small>")

    out = []
    for cur in order:
        _, mod = cm.info(cur, "__module__")
        anchor = re.sub(r"\W", "-", cur).lower()
        out += [f"### `{cur}` {{#{anchor}}}", "", f"{_cell(mod)} · [open file ↗]({cm.url}/{cur})", ""]
        out += ["| Function | Step | Calls | Called by | What it does |", "|---|---|---|---|---|"]
        for k, node in sorted(((k, n) for k, n in defs.items() if k[0] == cur), key=lambda kv: kv[1].lineno):
            if isinstance(node, ast.ClassDef) and not ast.get_docstring(node):
                continue
            n = cm.step_of.get(k)
            step = f"[{n}]({WALK}#step-{n})" if n else ""
            doc = (ast.get_docstring(node) or "").strip().splitlines()
            calls = ", ".join(ref(d) for d in sorted(graph.get(k, ()), key=lambda d: d[1]))
            by = ", ".join(ref(c) for c in sorted(callers.get(k, ()), key=lambda d: d[1]))
            out.append(
                f"| [`{k[1]}`]({cm.url}/{cur}#L{node.lineno}) | {step} | {calls or '—'} | {by or '—'} | {_cell(doc[0]) if doc else '—'} |"
            )
        out.append("")
    return "\n".join(out)


def on_page_markdown(markdown, page, config, files):
    if "{{" not in markdown:
        return markdown
    if HARDCODED.search(markdown):
        url = yaml.safe_load(SPEC.read_text())["repo_url"].rstrip("/")
        markdown = HARDCODED.sub(lambda m: hardcoded_table(url), markdown)
    if "codemap" in markdown or TOUR_RE.search(markdown) or FILES_RE.search(markdown):
        cm = CodeMap()
        markdown = MARKER.sub(lambda m: cm.render(m.group(1)), markdown)
        markdown = TOUR_RE.sub(lambda m: cm.tour_page(), markdown)
        markdown = FILES_RE.sub(lambda m: files_page(cm), markdown)
    return markdown
