"""Build + execute the notebooks in ./Notebook from the percent-format sources in ./notebook_src.

    uv run python scripts/build_notebooks.py              # everything (4 parallel jobs)
    uv run python scripts/build_notebooks.py 01 07 -j 2   # selected notebooks
    uv run python scripts/build_notebooks.py --no-exec    # only (re)write the .ipynb files

Each source is a plain .py file with `# %%` (code) and `# %% [md]` (markdown) cells. The builder wraps every notebook with
a consistent banner, table of contents, numbered sections, key takeaways, references and navigation.
"""
import argparse, os, re, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
SRC, OUT = ROOT / "notebook_src", ROOT / "Notebook"
sys.path.insert(0, str(Path(__file__).parent))
from notebook_meta import META  # noqa: E402


def parse(text):
    cells, kind, buf = [], None, []
    def flush():
        lines = [re.sub(r"^# ?", "", l) for l in buf] if kind == "md" else buf   # md lines carry the '# ' comment prefix
        body = "\n".join(lines).strip("\n")
        if kind and body:
            cells.append((nbformat.v4.new_markdown_cell if kind == "md" else nbformat.v4.new_code_cell)(body))
    for line in text.splitlines():
        m = re.match(r"# %%\s*(\[md\])?", line)
        if m:
            flush(); kind = "md" if m.group(1) else "code"; buf = []
        else:
            buf.append(line)
    flush(); return cells


def number_sections(cells):
    """'## Foo' -> '## 1 · Foo' (skipping the title cell) and return [(n, title)]."""
    toc, n = [], 0
    for c in cells[1:]:
        if c.cell_type != "markdown":
            continue
        lines = c.source.split("\n")
        for i, l in enumerate(lines):
            if l.startswith("## ") and not re.match(r"## \d+ ·", l):
                n += 1; title = l[3:].strip(); lines[i] = f"## {n} · {title}"; toc.append((n, title))
        c.source = "\n".join(lines)
    return toc


def anchor(n, title):
    from urllib.parse import quote
    return quote(f"{n}-·-" + re.sub(r"\s+", "-", title))


def banner(meta, runtime, toc, nb_id):
    rows = [("Family", meta["family"]), ("Representative models", meta["models"]), ("Core idea", meta["idea"]),
            ("Compute / memory", meta["cost"]), ("Dataset", meta.get("data", "tiny-Shakespeare (char level) – see notebook 00")),
            ("Runtime", f"≈ {runtime:.0f} s on a laptop CPU" if runtime else "–")]
    md = "---\n\n### At a glance\n\n| | |\n|---|---|\n" + "\n".join(f"| **{k}** | {v} |" for k, v in rows)
    if meta.get("objectives"):
        md += "\n\n### What you will learn\n\n" + "\n".join(f"- {o}" for o in meta["objectives"])
    if toc:
        md += "\n\n### Contents\n\n" + "\n".join(f"{n}. [{t}](#{anchor(n, t)})" for n, t in toc)
    md += "\n\n> Environment: `uv sync` then `uv run jupyter lab` (kernel **LLM Architecture (uv)**). Data: `uv run python scripts/prepare_data.py`.\n\n---"
    return nbformat.v4.new_markdown_cell(md)


def footer(meta, stem, order):
    md = "---\n\n## Key takeaways\n\n" + "\n".join(f"- {t}" for t in meta.get("takeaways", []))
    if meta.get("refs"):
        md += "\n\n## References\n\n" + "\n".join(f"- {r}" for r in meta["refs"])
    i = order.index(stem); nav = []
    if i > 0: nav.append(f"← previous: `{order[i-1]}`")
    if i < len(order) - 1: nav.append(f"next: `{order[i+1]}` →")
    return nbformat.v4.new_markdown_cell(md + "\n\n---\n*" + "  ·  ".join(nav) + "*")


def refresh_md(stem):
    """Rewrite only the markdown cells (banner, TOC, text, footer) of an already executed notebook; outputs are kept."""
    meta = META.get(stem, {}); order = sorted(p.stem for p in SRC.glob("*.py"))
    old = nbformat.read(OUT / f"{stem}.ipynb", 4)
    new = parse((SRC / f"{stem}.py").read_text()); toc = number_sections(new)
    body = old.cells[2:-1]
    assert len(body) == len(new) - 1 and all(a.cell_type == b.cell_type for a, b in zip(body, new[1:])), f"{stem}: structure changed, re-execute"
    merged = [n if n.cell_type == "markdown" else o for o, n in zip(body, new[1:])]
    m = re.search(r"≈ (\d+) s", old.cells[1].source)
    old.cells = [new[0], banner(meta, float(m.group(1)) if m else None, toc, stem)] + merged + [footer(meta, stem, order)]
    nbformat.write(old, OUT / f"{stem}.ipynb")
    return f"{stem:<36} markdown refreshed"


def run(stem, execute=True):
    meta = META.get(stem, {}); order = sorted(p.stem for p in SRC.glob("*.py"))
    cells = parse((SRC / f"{stem}.py").read_text()); toc = number_sections(cells)
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = dict(name="llm-architecture", display_name="LLM Architecture (uv)", language="python")
    t0 = time.time(); status = "built"
    if execute:
        try:
            NotebookClient(nb, timeout=3600, kernel_name="llm-architecture", resources={"metadata": {"path": str(OUT)}}).execute(); status = "executed OK"
        except Exception as e:  # keep the partial notebook for debugging
            status = "FAILED: " + str(e)[-1200:]
    rt = time.time() - t0 if execute else None
    for c in nb.cells:                                        # tidy outputs: drop stderr-only noise
        if c.cell_type == "code":
            c.outputs = [o for o in c.get("outputs", []) if not (o.output_type == "stream" and o.name == "stderr")]
    nb.cells = [nb.cells[0], banner(meta, rt, toc, stem)] + nb.cells[1:] + [footer(meta, stem, order)]
    OUT.mkdir(exist_ok=True); nbformat.write(nb, OUT / f"{stem}.ipynb")
    return f"{stem:<36} {status} ({time.time()-t0:.0f}s)"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("names", nargs="*"); ap.add_argument("-j", "--jobs", type=int, default=4); ap.add_argument("--no-exec", action="store_true"); ap.add_argument("--md-only", action="store_true", help="refresh markdown of executed notebooks, keep outputs")
    a = ap.parse_args()
    stems = [p.stem for p in sorted(SRC.glob("*.py")) if not a.names or any(n in p.stem for n in a.names)]
    os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
    if a.md_only:
        for st in stems: print(refresh_md(st))
        sys.exit(0)
    # slowest first so the pool stays busy
    stems.sort(key=lambda s: s not in ("09_state_space_mamba", "11_hybrid_jamba", "07_mixture_of_experts", "05_attention_variants"))
    with ProcessPoolExecutor(a.jobs) as ex:
        for line in ex.map(run, stems, [not a.no_exec] * len(stems)):
            print(line, flush=True)
