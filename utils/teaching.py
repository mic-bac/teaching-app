"""
teaching.py
-----------
UI-free helpers that support the *teaching* aspects of the app: showing learners
the real code that runs, reusing the sibling ``postgresql/`` repo's setup command
instead of duplicating it, and describing the runtime architecture as a diagram.

Kept free of Streamlit so it can be unit-tested directly. The pages import these
and do the actual ``st.*`` rendering themselves.
"""

import inspect
import re
from pathlib import Path


def source_of(obj) -> str:
    """Return the source code of a function/class as a string.

    Used to display the *actual executed* implementation in the app, so the code
    a learner sees can never drift out of sync with what really runs.
    """
    return inspect.getsource(obj)


_SECTION_RE = re.compile(r"^#\s*---\s*(\d+)\.\s*(.+?)\s*---\s*$")


def split_script_blocks(source: str) -> list[dict]:
    """Split a ``parallel.*`` script into ordered sections by its step markers.

    Both ``parallel/parallel.py`` and ``parallel/parallel.R`` annotate each step
    with a matching ``# --- N. Title ---`` comment. This extracts the code under
    each marker so the two languages can be shown block-by-block, side by side,
    with equivalent steps lined up.

    A section's code runs from just after its marker to the next section marker
    or notebook cell boundary (a ``# %%`` line in the Python file), with the
    surrounding blank lines trimmed. Returns a list of
    ``{"number": int, "title": str, "code": str}`` dicts in file order.
    """
    lines = source.splitlines()
    markers = [(i, m) for i, line in enumerate(lines) if (m := _SECTION_RE.match(line))]

    blocks = []
    for idx, (line_no, match) in enumerate(markers):
        start = line_no + 1
        end = markers[idx + 1][0] if idx + 1 < len(markers) else len(lines)

        body = []
        for line in lines[start:end]:
            if line.lstrip().startswith("# %%"):
                break  # a notebook cell boundary ends this section's code
            body.append(line)

        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()

        blocks.append(
            {
                "number": int(match.group(1)),
                "title": match.group(2).strip(),
                "code": "\n".join(body),
            }
        )
    return blocks


def postgres_docker_command(readme_path: Path | None = None) -> str:
    """Extract the ``docker run`` command from the sibling ``postgresql`` repo.

    Reads ``postgresql/README.md`` and returns the first fenced ``bash`` block that
    contains ``docker run``. This keeps the setup command in a single source of
    truth (the postgresql guide) instead of duplicating it in the app.

    Falls back to a short pointer if the README or block can't be found.
    """
    if readme_path is None:
        readme_path = Path(__file__).resolve().parents[1] / "postgresql" / "README.md"

    fallback = "# See postgresql/README.md for the full 'docker run' command."
    try:
        text = readme_path.read_text(encoding="utf-8")
    except OSError:
        return fallback

    # Walk the ```bash ... ``` fenced blocks and return the first with `docker run`.
    parts = text.split("```")
    for i in range(1, len(parts), 2):  # odd indices are the fenced contents
        block = parts[i]
        first_nl = block.find("\n")
        lang = block[:first_nl].strip().lower()
        body = block[first_nl + 1 :] if first_nl != -1 else ""
        if lang == "bash" and "docker run" in body:
            return body.strip()
    return fallback


def architecture_dot(active_backend: str) -> str:
    """Return a Graphviz DOT string describing the app's runtime architecture.

    ``st.graphviz_chart`` renders a raw DOT string client-side, so this needs no
    Graphviz Python package. The node matching ``active_backend`` (``"PostgreSQL"``
    or anything else → the SQLite fallback) is highlighted so learners can see
    which backend is live right now.
    """
    pg_active = active_backend == "PostgreSQL"

    pg_fill = "#c8e6c9" if pg_active else "#ffffff"
    pg_pen = "3" if pg_active else "1"
    sqlite_fill = "#ffe0b2" if not pg_active else "#ffffff"
    sqlite_pen = "3" if not pg_active else "1"

    return f"""
digraph architecture {{
    rankdir=LR;
    bgcolor="transparent";
    node [fontname="Helvetica", shape=box, style="rounded,filled", fillcolor="#ffffff"];
    edge [fontname="Helvetica", fontsize=10];

    browser  [label="🌐 Your browser\\n(Streamlit UI)"];
    app      [label="🐍 Python app\\nSQLAlchemy engine", fillcolor="#e3f2fd"];
    postgres [label="🐘 PostgreSQL\\nDocker container\\nlocalhost:5432", fillcolor="{pg_fill}", penwidth={pg_pen}];
    sqlite   [label="📄 SQLite fallback\\ndata/fallback.db", fillcolor="{sqlite_fill}", penwidth={sqlite_pen}];

    browser -> app    [label="HTTP :8501"];
    app -> postgres   [label="try first\\npsycopg2"];
    app -> sqlite     [label="fallback if\\nunreachable", style=dashed];
}}
""".strip()
